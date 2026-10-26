# %% [markdown]
# # Eksperimen split sampel 70/30: baseline vs GWO
# CSV pembimbing → eksplorasi data → split sampel 70/30 → baseline → sembilan skenario
# GWO → pemilihan dari CV train → evaluasi test → tabel dan grafik.
#
# Fitur dan label hanya dari CSV recurrence top2000 di datasetv2; ID pasien tidak tersedia.
# Normalisasi sumber dan seleksi top2000 belum terverifikasi.
# CV merupakan skor internal optimasi; prediksi 0/1 bukan probabilitas klinis.
#
# Notebook ini menggunakan seed 42 untuk split sampel dan optimizer.
# Konfigurasi ini berasal dari pencarian TEST sebelumnya: hasil tetap eksploratif.
# S9 (Polynomial, pop30) tersedia; pemenang CV tetap ditentukan dari train, tidak dipaksa S9.
#
# CSV top2000 difilter menjadi top500 varians train; tanpa metadata pasien tambahan.

# %%
from pathlib import Path
from datetime import datetime
import csv
import os
import re
import hashlib
import json
import time
import sys
import argparse

parser = argparse.ArgumentParser(description='Run the single-split E20 experiment reported in the paper.')
parser.add_argument('--data', type=Path, help='Path to the study input CSV; defaults to data/GSE21510_recurrence_top2000.csv.')
parser.add_argument('--output', type=Path, help='Output parent directory; defaults to outputs/.')
args = parser.parse_args() if '__file__' in globals() and 'ipykernel' not in sys.modules else parser.parse_args([])

import joblib
import numpy as np
import pandas as pd
from IPython import get_ipython
from IPython.display import display
import matplotlib
if get_ipython() is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, classification_report)
from niapy.algorithms.basic import GreyWolfOptimizer
from niapy.callbacks import Callback
from niapy.problems import Problem
from niapy.task import Task

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
TOP_VARIANCE_FEATURES = 500
SEED = 42
TEST_SIZE = 0.30
CV_FOLDS = 10
KERNELS = ('linear', 'rbf', 'poly')
POPULATIONS = (10, 20, 30)
ITERATIONS = 40
N_JOBS = 1
LABELS = ['No recurrence', 'Recurrence']
METRICS = ('Acc', 'BA', 'Prec', 'Rec', 'F1')


def find_project():
    location = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    for folder in (location, *location.parents):
        if (folder / 'experiments/e20.py').is_file() and (folder / 'README.md').is_file():
            return folder
    raise FileNotFoundError('Run from this repository or its experiments folder.')


BASE_DIR = find_project()
DATA_PATH = args.data.expanduser().resolve() if args.data else BASE_DIR / 'data/GSE21510_recurrence_top2000.csv'
if not DATA_PATH.is_file():
    raise FileNotFoundError(f'Input CSV not found: {DATA_PATH}. See data/README.md or use --data PATH.')
OUTPUT_DIR = args.output.expanduser().resolve() if args.output else BASE_DIR / 'outputs'

RATIOS = (0.30,)  # 70% train / 30% test pada sampel GSM
SPLIT_SEEDS = (42,)

# %% [markdown]
# ## Persiapan output eksperimen

# %%
# Seed split sampel ditetapkan 42.
split_seed = 42
test_fraction = 0.30
study_dir = OUTPUT_DIR / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
RUN_DIR = Path(study_dir) / f'ratio_{int((1-test_fraction)*100)}_seed_{split_seed}'
TEST_SIZE = test_fraction
for name in ('reports', 'plots', 'splits', 'extracts', 'models', 'paper_outputs'):
    (RUN_DIR / name).mkdir(parents=True, exist_ok=True)
def table(title, frame, filename, folder='reports', show=True):
    """CSV menyimpan presisi asli; tampilan ringkas tidak membulatkan parameter kecil ke nol."""
    frame.to_csv(RUN_DIR / folder / f'{filename}.csv', index=False)
    if show:
        print(f'\n{title}')
        if get_ipython() is not None:
            display(frame)
        else:
            print(frame.to_string(index=False, float_format=lambda value: f'{value:.4g}'))
    return frame
def save_plot(fig, name, folder='plots'):
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(RUN_DIR / folder / f'{name}.{extension}', dpi=180, bbox_inches='tight')
    if get_ipython() is not None:
        display(fig)
    plt.close(fig)
print(f'Dataset: {DATA_PATH}\nOutput: {RUN_DIR}')
print('ROS aktif dalam pipeline SVM; validation/test tetap asli. CV tetap internal/post-selection.')

# %% [markdown]
# ## Mengenal dataset
# Label recurrence: 0 = tidak kambuh, 1 = kambuh. Stage ditampilkan sebagai metadata,
# tetapi tidak menjadi prediktor. Kolom GSM dan recurrence juga dikeluarkan dari X.
# EDA bersifat deskriptif: tidak digunakan memilih fitur, split, atau parameter.

# %%
with DATA_PATH.open(encoding='utf-8-sig', newline='') as source:
    header = next(csv.reader(source))
if len(header) != len(set(header)):
    raise ValueError('Nama kolom dataset duplikat.')
data = pd.read_csv(DATA_PATH, dtype={header[0]: str})
id_column = data.columns[0]
sample_ids = data[id_column].str.strip()
if sample_ids.isna().any() or sample_ids.duplicated().any() or not sample_ids.str.fullmatch(r'GSM\d+').all():
    raise ValueError('ID GSM tidak valid, kosong, atau duplikat.')
if not {'recurrence', 'stage'}.issubset(data):
    raise ValueError('Kolom recurrence dan stage wajib tersedia.')
y = pd.to_numeric(data['recurrence'], errors='raise').to_numpy()
if set(np.unique(y)) != {0, 1}:
    raise ValueError('Target harus berisi dua kelas: 0 dan 1.')
y = y.astype(int)
features = data.drop(columns=[id_column, 'recurrence', 'stage']).astype(np.float32)
if features.shape[1] != 2000 or not np.isfinite(features.to_numpy()).all():
    raise ValueError('Diperlukan 2000 probe numerik finite; tidak ada imputasi global.')
X = features.to_numpy()
n_features = X.shape[1]
feature_names = features.columns.to_numpy()
print(f'{len(data)} sampel; {n_features:,} probe; label dan stage tidak masuk fitur.')
preview = data.rename(columns={id_column: 'sample_id'}).iloc[:5, :9]
_ = table('Contoh baris dan probe pertama', preview, 'eda_preview')
class_counts = pd.Series(y).value_counts().sort_index()
_ = table('Distribusi target', pd.DataFrame({'Kelas': LABELS, 'Jumlah': class_counts.values,
      'Persentase': class_counts.values / len(y) * 100}), 'eda_classes')
_ = table('Stage dan label (metadata saja)', pd.crosstab(data['stage'].fillna('Missing'),
      data['recurrence']).reset_index(), 'eda_stage')
feature_summary = features.describe().T.reset_index(names='Probe')
_ = table('Statistik contoh probe (seluruh probe tersimpan di CSV)', feature_summary.head(10), 'eda_probe_preview')
_ = table('', feature_summary, 'eda_all_probes', show=False)
quality = pd.DataFrame({'Pemeriksaan': ['ID duplikat', 'Nilai fitur non-finite',
    'Profil ekspresi duplikat', 'Probe konstan'], 'Jumlah': [int(sample_ids.duplicated().sum()),
    int((~np.isfinite(X)).sum()), int(features.duplicated().sum()), int(features.nunique().eq(1).sum())]})
_ = table('Pemeriksaan kualitas', quality, 'eda_quality')
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
bars = axes[0].bar(LABELS, class_counts.values, color=['#56799c', '#bd6666'])
axes[0].bar_label(bars, padding=3)
axes[0].set(ylabel='Samples', ylim=(0, class_counts.max() * 1.2))
axes[1].hist(X.ravel(), bins=50, color='#56799c')
axes[1].set(xlabel='Provided expression value', ylabel='Frequency')
save_plot(fig, 'eda_distribution')
print('Batas: asal filter top2000 dan independensi normalisasi belum terverifikasi.')

# %% [markdown]
# ## Pemisahan train dan test
# Satu split stratifikasi sampel dengan seed tetap (70/30 sampel); test tidak digunakan untuk pemilihan model.
# Sepuluh fold hanya pada train. Tidak ada log2 atau normalisasi tambahan. Setelah split, pilih top500 varians dari train; probe yang sama diterapkan pada test.
#
# Pemisahan berdasarkan GSM. Independensi pasien tidak dapat diverifikasi dari CSV ini.

# %%
# CSV berisi ID sampel GSM; identitas pasien tidak tersedia.
groups = sample_ids.to_numpy()
train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=TEST_SIZE,
                                      stratify=y, random_state=split_seed)
# Ranking dipelajari dari train saja, tanpa nilai atau label test.
source_feature_count = features.shape[1]
train_variance = features.iloc[train_idx].astype(np.float64).var(axis=0, ddof=0)
ranking = train_variance.sort_values(ascending=False, kind='stable')
selected_probes = ranking.head(TOP_VARIANCE_FEATURES).index
variance_report = ranking.rename('train_variance').rename_axis('Probe').reset_index()
variance_report['selected'] = variance_report['Probe'].isin(selected_probes)
_ = table('Contoh ranking varians train', variance_report.head(10), 'variance_preview')
_ = table('', variance_report, 'train_variance_ranking', show=False)
features = features.loc[:, selected_probes].copy()
X = features.to_numpy()
n_features = features.shape[1]
feature_names = features.columns.to_numpy()
assert n_features == TOP_VARIANCE_FEATURES
print(f'Filter varians train: {source_feature_count} → {n_features} probe.')
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train, groups_test = groups[train_idx], groups[test_idx]
if np.bincount(y_train, minlength=2).min() < CV_FOLDS:
    raise ValueError('Sampel kelas minoritas train tidak cukup untuk CV.')
cv_seed = split_seed
cv_splits = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=cv_seed).split(X_train, y_train))
print(f'Holdout sampel dan {CV_FOLDS}-fold CV; seed={cv_seed}. Identitas pasien tidak tersedia.')

def check_partition(values, target, left, right):
    if not np.array_equal(np.sort(np.r_[left, right]), np.arange(len(target))):
        raise ValueError('Split harus mencakup sampel tepat sekali.')
    if any(set(np.unique(target[idx])) != {0, 1} for idx in (left, right)):
        raise ValueError('Setiap subset harus memiliki kedua kelas.')
    # Bandingkan byte profil float32: tidak menyamakan ID sampel dengan pasien.
    if {row.tobytes() for row in values[left]} & {row.tobytes() for row in values[right]}:
        raise ValueError('Profil ekspresi identik melintasi batas split.')
check_partition(X, y, train_idx, test_idx)
for left, right in cv_splits:
    check_partition(X_train, y_train, left, right)
split_manifest = pd.DataFrame({'sample_id': sample_ids, 'label': y, 'split': 'train'})
split_manifest.loc[test_idx, 'split'] = 'test'
_ = table('', split_manifest, 'sample_split', 'splits', False)
fold_manifest = split_manifest.iloc[train_idx].copy().reset_index(drop=True)
for fold, (_, valid) in enumerate(cv_splits, 1):
    fold_manifest.loc[valid, 'validation_fold'] = fold
fold_manifest['validation_fold'] = fold_manifest['validation_fold'].astype(int)
_ = table('', fold_manifest, 'cv_sample_folds', 'splits', False)
for name, indices in [('train', train_idx), ('test', test_idx)]:
    features.iloc[indices].set_axis(sample_ids.iloc[indices]).to_csv(RUN_DIR / 'extracts' / f'X_{name}.csv')
    pd.DataFrame({'sample_id': sample_ids.iloc[indices], 'recurrence': y[indices]}).to_csv(
        RUN_DIR / 'extracts' / f'y_{name}.csv', index=False)
split_table = pd.DataFrame({'Kelas': LABELS, 'Train': np.bincount(y_train, minlength=2),
    'Test': np.bincount(y_test, minlength=2), 'Total': np.bincount(y, minlength=2)})
split_table.loc[len(split_table)] = ['Total sampel', len(y_train), len(y_test), len(y)]
_ = table('Komposisi train dan test', split_table, 'split_summary')
audit = dict(source=str(DATA_PATH), sha256=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
    split_unit='sample', cv_seed=cv_seed, seed=split_seed, optimizer_seed=SEED, test_fraction=TEST_SIZE, cv_folds=CV_FOLDS,
    excluded_predictors=[str(id_column), 'recurrence', 'stage'], sample_overlap=0,
    cross_split_duplicate_profiles=0, patient_independence_verified=False,
    source_filter_independence_verified=False, source_normalization_independence_verified=False,
    overall_leakage_free_claim=False, selection_metric='mean CV BA/F1', optimizer_iterations=ITERATIONS, gwo_evaluation_budget='population * (iterations + 1)', configuration_chosen_from_prior_test_search=True, nested_cv=False,
    resampling='RandomOverSampler inside every SVM fit (GWO fitness, grid CV, final CV, final train)',
    validation_test_resampled=False, sampler_seed=SEED,
    cv_scope='internal/post-selection', transformations='top500 variance fitted on outer train; expression values unchanged',
    additional_variance_filter=True, variance_selection_scope='outer train; CV post-selection', variance_ddof=0, source_features=source_feature_count, selected_features=n_features)
(RUN_DIR / 'reports/leakage_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')

# %% [markdown]
# ## Evaluasi dengan random oversampling dan pencarian parameter
# Ringkasan ditampilkan per model. Rincian fold tetap disimpan ke CSV tanpa memenuhi layar.
# Tuning dan GWO mengoptimalkan rata-rata CV BA/F1.
# Dummy uniform tetap pada data asli. BA/F1 memiliki bobot sama; F1 untuk kelas kambuh.

# %%
def make_svm(kernel):
    # Pipeline.fit melakukan sampling hanya pada data yang diberikan oleh fold train.
    # Pipeline.predict melewati sampler: validation/test tidak pernah digandakan.
    return Pipeline([
        ('ros', RandomOverSampler(sampling_strategy='auto', random_state=SEED)),
        ('svm', SVC(kernel=kernel, random_state=SEED)),
    ])

def metrics(target, predicted):
    scores = [accuracy_score(target, predicted), balanced_accuracy_score(target, predicted),
              precision_score(target, predicted, zero_division=0),
              recall_score(target, predicted, zero_division=0), f1_score(target, predicted, zero_division=0)]
    result = dict(zip(METRICS, scores))
    result.update(zip(('TN', 'FP', 'FN', 'TP'), confusion_matrix(target, predicted, labels=[0, 1]).ravel().tolist()))
    return result
def evaluate_cv(estimator, values, name):
    rows = []
    def scorer(model, valid_values, target):
        row = metrics(target, model.predict(valid_values))
        rows.append({'Model': name, 'Fold': len(rows) + 1, **row})
        return row['Acc']
    scores = cross_val_score(estimator, values, y_train, cv=cv_splits, scoring=scorer,
                             n_jobs=1, error_score='raise')
    if len(rows) != CV_FOLDS or not np.allclose(scores, [row['Acc'] for row in rows]):
        raise RuntimeError('Pencatatan fold CV tidak konsisten.')
    frame = pd.DataFrame(rows)
    table('', frame, f'cv_{name}', show=False)
    return frame
def parameter_grid(kernel):
    grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    if kernel != 'linear':
        grid['gamma'] = [1, 0.1, 0.01, 0.001, 0.0001]
    if kernel == 'poly':
        grid['degree'] = [2, 3, 4, 5]
    return grid
def joint_score(model, values, target):
    prediction = model.predict(values)
    return (balanced_accuracy_score(target, prediction) +
            f1_score(target, prediction, zero_division=0)) / 2


def tune(kernel, values, name):
    search = GridSearchCV(make_svm(kernel), {f'svm__{key}': value for key, value in parameter_grid(kernel).items()},
        scoring={'accuracy': 'accuracy', 'balanced_accuracy': 'balanced_accuracy',
                 'f1': 'f1', 'joint': joint_score},
        refit='joint', cv=cv_splits, n_jobs=N_JOBS, error_score='raise', verbose=0)
    search.fit(values, y_train)
    table('', pd.DataFrame(search.cv_results_), f'grid_{name}', show=False)
    return search
def fit_candidate(name, kernel, mask, population=None, history=None):
    started = time.perf_counter()
    values = X_train[:, mask]
    default = make_svm(kernel)
    default_cv = evaluate_cv(default, values, name)
    default.fit(values, y_train)
    search = tune(kernel, values, name)
    best = search.best_index_
    result = dict(id=name, kernel=kernel, mask=mask, population=population,
        n_features=int(mask.sum()), default=default, tuned=search.best_estimator_,
        cv_default=default_cv, cv_score=float(search.cv_results_['mean_test_accuracy'][best]),
        selection_score=float(search.best_score_),
        cv_f1=float(search.cv_results_['mean_test_f1'][best]),
        cv_std=float(search.cv_results_['std_test_accuracy'][best]),
        cv_ba=float(search.cv_results_['mean_test_balanced_accuracy'][best]),
        params={key.removeprefix('svm__'): value for key, value in search.best_params_.items()}, history=history, seconds=time.perf_counter() - started)
    print(f'[{name} {kernel.upper()}] CV default={default_cv.Acc.mean():.2%}; '
          f'tuned={result["cv_score"]:.2%}; BA tuned={result["cv_ba"]:.2%}; '
          f'F1 tuned={result["cv_f1"]:.2%}; joint={result["selection_score"]:.2%}; '
          f'fitur={mask.sum():,}; {result["seconds"]:.1f}s', flush=True)
    return result

# %% [markdown]
# ## Baseline
# Dummy uniform sebagai pembanding acak; SVM linear, RBF, dan polynomial tanpa seleksi fitur.

# %%
dummy = DummyClassifier(strategy='uniform', random_state=SEED)
dummy_cv = evaluate_cv(dummy, X_train, 'Dummy')
dummy.fit(X_train, y_train)
print(f'[Dummy uniform] CV accuracy={dummy_cv.Acc.mean():.2%}; BA={dummy_cv.BA.mean():.2%}')
baselines = {}
for kernel in KERNELS:
    print(f'[Baseline {kernel.upper()}] Evaluasi default dan grid search...', flush=True)
    baselines[kernel] = fit_candidate(f'Baseline_{kernel}', kernel, np.ones(n_features, dtype=bool))

# %% [markdown]
# ## Seleksi fitur GWO
# Fitness = 0.99 × (1 − rata-rata CV BA/F1) + 0.01 × proporsi fitur.
# Mask memakai x > 0.5, tanpa sigmoid. Progres satu baris per iterasi, tanpa log kandidat/fold.

# %%
class FeatureProblem(Problem):
    def __init__(self, kernel):
        super().__init__(dimension=n_features, lower=0.0, upper=1.0)
        self.kernel = kernel

    def _evaluate(self, position):
        mask = position > 0.5
        if not mask.any():
            return 1.0
        scores = cross_val_score(make_svm(self.kernel), X_train[:, mask],
                                 y_train, cv=cv_splits, scoring=joint_score, n_jobs=N_JOBS,
                                 error_score='raise')
        return 0.99 * (1 - scores.mean()) + 0.01 * mask.mean()
class History(Callback):
    def __init__(self, name):
        super().__init__()
        self.name, self.rows = name, []

    def after_iteration(self, population, fitness, best_x, best_fitness, **kwargs):
        selected = int((best_x > 0.5).sum())
        acc = 1 - (float(best_fitness) - 0.01 * selected / n_features) / 0.99
        row = dict(iteration=len(self.rows) + 1, best_fitness=float(best_fitness),
                   joint_score=acc, loss=1 - acc, n_selected=selected)
        self.rows.append(row)
        print(f'[{self.name}] Iter {len(self.rows):02}/{ITERATIONS} | '
              f'fitness={best_fitness:.5f} | CV BA/F1={acc:.2%} | fitur={selected}', flush=True)

# %% [markdown]
# ## Sembilan skenario
# Setiap kernel diuji dengan populasi 10, 20, dan 30. Semua skenario memakai 40 iterasi.
# Subset akhir GWO dipakai untuk evaluasi setiap skenario.
# Setelah GWO: evaluasi subset default dan tuning ulang, seluruhnya menggunakan train.

# %%
scenarios = []
for kernel in KERNELS:
    for population in POPULATIONS:
        name = f'S{len(scenarios) + 1}'
        print(f'\n[{name}] {kernel.upper()} | populasi={population} | mulai GWO', flush=True)
        started = time.perf_counter()
        history = History(name)
        optimizer = GreyWolfOptimizer(population_size=population, seed=SEED, callbacks=[history])
        # NiaPy evaluates the initial population once, then each population per iteration.
        # A finite budget makes a = 2 - 2 * evals / max_evals decrease toward zero.
        max_evaluations = population * (ITERATIONS + 1)
        task = Task(problem=FeatureProblem(kernel), max_iters=ITERATIONS,
                    max_evals=max_evaluations)
        print(f'[{name}] GWO: {ITERATIONS} iterasi | batas evaluasi={max_evaluations} | '
              f'a awal={2 - 2 * population / max_evaluations:.4f}, '
              f'a iterasi terakhir={2 / (ITERATIONS + 1):.4f}', flush=True)
        position, fitness = optimizer.run(task)
        mask = position > 0.5
        if not mask.any() or not np.isfinite(fitness):
            raise RuntimeError(f'{name}: subset GWO tidak valid.')
        result = fit_candidate(name, kernel, mask, population, pd.DataFrame(history.rows))
        result['seconds'] = time.perf_counter() - started
        result['fitness'] = float(fitness)
        scenarios.append(result)
        table('', result['history'], f'history_{name}', show=False)
        table('', pd.DataFrame({'Probe': feature_names[mask]}), f'features_{name}', show=False)
        print(f'[{name}] Selesai {result["seconds"]:.1f}s; parameter={result["params"]}', flush=True)

# %% [markdown]
# ## Pemilihan model dan evaluasi holdout
# Urutan pemilihan: Rata-rata CV BA/F1 presisi penuh, fitur lebih sedikit, kemudian ID skenario.
# Pilihan dikunci sebelum prediksi test. Seluruh skenario dilaporkan tanpa memilih ulang dari test.

# %%
def selection_key(result):
    return (-result['selection_score'], result['n_features'], int(result['id'][1:]))
winner = min(scenarios, key=selection_key)
representatives = {kernel: min([r for r in scenarios if r['kernel'] == kernel], key=selection_key)
                   for kernel in KERNELS}
lock = dict(winner=winner['id'], kernel=winner['kernel'], cv_accuracy=winner['cv_score'],
            cv_joint=winner['selection_score'],
            optimizer_iterations=ITERATIONS, gwo_evaluation_budget='population * (iterations + 1)', configuration_chosen_from_prior_test_search=True,
            features=winner['n_features'], parameters=winner['params'],
            rule='CV BA/F1 mean descending, feature count ascending, scenario ID ascending',
            representatives={k: r['id'] for k, r in representatives.items()}, test_used_for_selection=False)
(RUN_DIR / 'reports/selection_lock.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
print(f'Model terpilih dari train: {winner["id"]} {winner["kernel"].upper()}, {winner["n_features"]} probe.')
print('Aturan: rata-rata CV BA/F1 tertinggi; jika seri, fitur lebih sedikit lalu ID terkecil. Ini bukan pemenang berdasarkan test.')
sampling_rows = []
for record in [*baselines.values(), *scenarios]:
    for mode in ('default', 'tuned'):
        fitted = record[mode]
        indices = fitted.named_steps['ros'].sample_indices_
        counts = np.bincount(y_train[indices], minlength=2)
        assert counts[0] == counts[1]
        sampling_rows.append({'Model': record['id'], 'Mode': mode,
            'Original train rows': len(y_train), 'Resampled train rows': len(indices),
            'No recurrence': int(counts[0]), 'Recurrence': int(counts[1]),
            'Untouched test rows': len(y_test)})
_ = table('Audit random oversampling pada fit akhir', pd.DataFrame(sampling_rows), 'sampling_audit')
dummy_test = metrics(y_test, dummy.predict(X_test))
for result in [*baselines.values(), *scenarios]:
    values = X_test[:, result['mask']]
    for mode in ('default', 'tuned'):
        prediction = result[mode].predict(values)
        result[f'test_{mode}'] = metrics(y_test, prediction)
        table('', pd.DataFrame({'sample_id': sample_ids.iloc[test_idx].to_numpy(),
              'actual': y_test, 'predicted': prediction}), f'predictions_{result["id"]}_{mode}', show=False)
    joblib.dump(dict(model=result['tuned'], features=feature_names[result['mask']],
        target='recurrence', label_mapping=dict(enumerate(LABELS)), audit=audit),
        RUN_DIR / 'models' / f'{result["id"]}.joblib')
def recap_row(result):
    return {'ID': result['id'], 'Kernel': result['kernel'].upper(),
            'Pop': result['population'], 'Fitur': result['n_features'],
            'CV Default (%)': result['cv_default'].Acc.mean() * 100,
            'CV BA Default (%)': result['cv_default'].BA.mean() * 100,
            'CV Tuned (%)': result['cv_score'] * 100, 'CV Std (pp)': result['cv_std'] * 100,
            'CV BA Tuned (%)': result['cv_ba'] * 100,
            'CV F1 Tuned (%)': result['cv_f1'] * 100,
            'CV Joint Tuned (%)': result['selection_score'] * 100,
            **{f'Test {m} (%)': result['test_tuned'][m] * 100 for m in METRICS},
            'Parameters': json.dumps(result['params']), 'Waktu (s)': result['seconds']}
recap = pd.DataFrame([{'ID': 'Dummy uniform', 'Kernel': '-', 'Fitur': n_features,
    'CV Default (%)': dummy_cv.Acc.mean() * 100, 'CV BA Default (%)': dummy_cv.BA.mean() * 100,
    **{f'Test {m} (%)': dummy_test[m] * 100 for m in METRICS}},
    *[recap_row(r) for r in [*baselines.values(), *scenarios]]])
_ = table('Rekap baseline dan GWO (hasil tuned; Dummy tanpa tuning)', recap, 'rekapitulasi')
base = baselines[winner['kernel']]
jobs = [('Dummy uniform', dummy, np.ones(n_features, dtype=bool), dummy_test),
        ('P1 Full/default', base['default'], base['mask'], base['test_default']),
        ('P2 Full/tuned', base['tuned'], base['mask'], base['test_tuned']),
        ('P3 GWO/default', winner['default'], winner['mask'], winner['test_default']),
        ('P4 GWO/tuned', winner['tuned'], winner['mask'], winner['test_tuned'])]
pipeline_table = pd.DataFrame([{'Model': name, 'Fitur': int(mask.sum()),
    **{f'Test {m} (%)': test[m] * 100 for m in METRICS},
    **{m: test[m] for m in ('TN', 'FP', 'FN', 'TP')}} for name, _, mask, test in jobs[1:]])
_ = table(f'Empat pipeline untuk kernel {winner["kernel"].upper()}', pipeline_table, 'komparasi_p1_p4_test')
print(classification_report(y_test, winner['tuned'].predict(X_test[:, winner['mask']]),
                            labels=[0, 1], target_names=LABELS, zero_division=0))

# %% [markdown]
# ## Validasi akhir: CV train dan test
# cross_val_score menguji ulang baseline dan seluruh skenario pada train. Mask tidak dipilih ulang
# di dalam fold: CV ini internal/post-selection, bukan nested CV atau evaluasi independen.
#
# Filter top500 dan subset GWO dipilih dari train; CV setelah seleksi tetap internal/post-selection, bukan nested CV.

# %%
# Semua model ditampilkan agar pemenang global tidak menutupi hasil skenario lain.
final_jobs = [('Dummy uniform', '-', 'Dummy', 'default', dummy,
               np.ones(n_features, dtype=bool), dummy_test)]
for result in [*baselines.values(), *scenarios]:
    for mode in ('default', 'tuned'):
        final_jobs.append((f"{result['id']} | {result['kernel'].upper()} | {mode}",
            result['kernel'], result['id'], mode, result[mode], result['mask'], result[f'test_{mode}']))
final_rows, final_folds = [], []
for index, (name, kernel, scenario_id, mode, estimator, mask, test) in enumerate(final_jobs):
    folds = evaluate_cv(clone(estimator), X_train[:, mask], f'Final_{index}')
    folds['Model'] = name
    final_folds.append(folds)
    row = {'Model': name, 'Kernel': kernel, 'ID': scenario_id, 'Mode': mode,
           'Fitur': int(mask.sum()), 'Pemenang global CV': scenario_id == winner['id']}
    for metric in METRICS:
        row.update({f'CV Train {metric} (%)': folds[metric].mean() * 100,
                    f'CV Train {metric} Std (pp)': folds[metric].std(ddof=0) * 100,
                    f'Test {metric} (%)': test[metric] * 100})
    final_rows.append(row)
    print(f'[{name}] CV={row["CV Train Acc (%)"]:.2f}%; test={row["Test Acc (%)"]:.2f}%; '
          f'BA test={row["Test BA (%)"]:.2f}%')
final_table = table('Validasi akhir semua model: CV internal train vs test',
                    pd.DataFrame(final_rows), 'final_cv_train_vs_test')
_ = table('', pd.concat(final_folds), 'final_cv_train_folds', show=False)

# Bandingkan setiap GWO tuned dengan baseline tuned dari kernel yang SAMA.
paired_rows = []
for result in scenarios:
    base_row = final_table[(final_table['ID'] == baselines[result['kernel']]['id']) &
                           (final_table['Mode'] == 'tuned')].iloc[0]
    gwo_row = final_table[(final_table['ID'] == result['id']) &
                          (final_table['Mode'] == 'tuned')].iloc[0]
    row = {'Skenario': result['id'], 'Kernel': result['kernel'], 'Populasi': result['population'],
           'Fitur': result['n_features'], 'Pemenang global CV': result['id'] == winner['id']}
    for scope in ('CV Train', 'Test'):
        for metric in ('Acc', 'BA'):
            column = f'{scope} {metric} (%)'
            row[f'Baseline {column}'] = base_row[column]
            row[f'GWO {column}'] = gwo_row[column]
            row[f'Delta {scope} {metric} (pp)'] = gwo_row[column] - base_row[column]
    row['CV Acc dan Test Acc meningkat'] = (row['Delta CV Train Acc (pp)'] > 1e-9 and
                                            row['Delta Test Acc (pp)'] > 1e-9)
    paired_rows.append(row)
final_paired_table = table('Semua skenario vs baseline kernel yang sama (tuned)',
                           pd.DataFrame(paired_rows), 'final_paired_cv_test_all_scenarios')
print('Tabel ini pelaporan seluruh skenario; tidak mengubah pilihan model berdasarkan test.')

# %% [markdown]
# ## Visualisasi hasil
# Confusion matrix, konvergensi, dinamika GWO, perbandingan tuning, dan reduksi fitur.

# %%
def draw_cm(ax, scores, title):
    cm = np.array([[scores['TN'], scores['FP']], [scores['FN'], scores['TP']]])
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=len(y_test))
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha='center', va='center',
                color='white' if value > len(y_test) / 2 else 'black')
    ax.set_xticks([0, 1], ['No\nrecurrence', 'Recurrence'])
    ax.set_yticks([0, 1], ['No\nrecurrence', 'Recurrence'])
    ax.set(title=title, xlabel='Predicted', ylabel='Actual')
for records, shape, name in [(list(baselines.values()), (1, 3), 'cm_gabungan_baseline'),
                            (scenarios, (3, 3), 'cm_gabungan_9scenarios')]:
    fig, axes = plt.subplots(*shape, figsize=(4 * shape[1], 3.6 * shape[0]), squeeze=False)
    for ax, result in zip(axes.flat, records):
        draw_cm(ax, result['test_tuned'], f'{result["id"]} {result["kernel"].upper()}')
    save_plot(fig, name)
def convergence(column, ylabel, filename, folder='plots'):
    fig, ax = plt.subplots(figsize=(10, 5))
    for result in scenarios:
        h = result['history']
        ax.plot(h.iteration, h[column], label=f'{result["id"]} {result["kernel"]} Pop{result["population"]}')
    ax.set(xlabel='Iteration', ylabel=ylabel)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    save_plot(fig, filename, folder)
convergence('best_fitness', 'Best fitness', 'gwo_9_convergence_comparison')
convergence('loss', 'Objective loss (1 - mean CV BA/F1)', 'gwo_9_loss_convergence_comparison')
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for ax, result in zip(axes.flat, scenarios):
    h = result['history']
    ax.plot(h.iteration, h.best_fitness, color='#bd6666')
    second = ax.twinx()
    second.plot(h.iteration, h.n_selected, '--', color='#56799c')
    ax.set(title=f'{result["id"]} {result["kernel"]} Pop{result["population"]}', xlabel='Iteration', ylabel='Fitness')
    second.set_ylabel('Selected probes')
save_plot(fig, 'gwo_dynamics_3x3_grid')
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(scenarios))
for offset, values, label in [(-.19, [r['cv_default'].Acc.mean() * 100 for r in scenarios], 'Default'),
                              (.19, [r['cv_score'] * 100 for r in scenarios], 'Tuned')]:
    bars = ax.bar(x + offset, values, .38, label=label)
    ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
ax.set_xticks(x, [r['id'] for r in scenarios])
ax.set(ylabel='Internal CV accuracy (%)', ylim=(0, 115))
ax.legend(loc='upper right')
save_plot(fig, 'cv_pre_vs_post_tuning')
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar([r['id'] for r in scenarios], [r['n_features'] for r in scenarios], color='#56799c')
ax.bar_label(bars, padding=4)
ax.set(ylabel='Selected probes', ylim=(0, max(r['n_features'] for r in scenarios) * 1.2))
save_plot(fig, 'feature_reduction')
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(['Initial', f'GWO {winner["id"]}'], [n_features, winner['n_features']], color=['#56799c', '#bd6666'])
ax.bar_label(bars, padding=4)
ax.set(ylabel='Probes', ylim=(0, n_features * 1.2))
save_plot(fig, 'feature_reduction_stages')

# %% [markdown]
# ## Graph dan table buat paper
# Empat tabel dan empat figure dari variabel run ini, tanpa angka hasil yang ditulis manual.
# Tabel yang sudah tampil di atas cukup diekspor kembali; tidak dicetak dua kali.

# %%
scenario_table = pd.DataFrame([{'ID': r['id'], 'Kernel': r['kernel'].upper(), 'Pop GWO': r['population'],
    'Fitur': r['n_features'], 'Waktu (s)': r['seconds'],
    'Reduksi (%)': (1 - r['n_features'] / n_features) * 100} for r in scenarios])
for title, frame, filename, show in [('', split_table, 'table_split', False),
    ('Ringkasan skenario dan reduksi fitur', scenario_table, 'table_scenarios', True),
    ('', final_table, 'table_cv_train_test', False), ('', recap, 'table_kernel_comparison', False)]:
    table(title, frame, filename, 'paper_outputs', show)
fig, ax = plt.subplots(figsize=(9, 4))
ax.axis('off')
flow = [f'CSV: {len(y)} samples, {n_features:,} probes',
        f'Sample split: {len(y_train)} train / {len(y_test)} test',
        f'Train: Dummy + {len(KERNELS)} baseline kernels + {len(scenarios)} GWO scenarios',
        f'{CV_FOLDS}-fold internal CV: selection and tuning', 'Locked selection → holdout evaluation']
for index, label in enumerate(flow):
    height = .9 - index * .2
    ax.text(.5, height, label, ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=.5', fc='white', ec='black'))
    if index < len(flow) - 1:
        ax.annotate('', xy=(.5, height - .14), xytext=(.5, height - .06),
                    xycoords='axes fraction', arrowprops=dict(arrowstyle='->'))
save_plot(fig, 'figure_workflow', 'paper_outputs')
counts = scenario_table.pivot(index='Pop GWO', columns='Kernel', values='Fitur').sort_index()
with plt.rc_context({'font.family': 'serif', 'font.size': 11}):
    fig, ax = plt.subplots(figsize=(6, 4.3))
    if counts.nunique(axis=1).eq(1).all():
        bars = ax.bar(counts.index.astype(str), counts.iloc[:, 0], width=.55,
                      color=['#56799c', '#375d7d', '#7199b8'])
        ax.bar_label(bars, labels=[f'{v:,}' for v in counts.iloc[:, 0]], padding=4)
    else:
        counts.plot.bar(ax=ax, rot=0, width=.75)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', rotation=90, padding=4, fontsize=9)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=len(KERNELS), frameon=False)
    ax.set(xlabel='GWO population', ylabel='Selected probes', ylim=(0, counts.max().max() * 1.35))
    ax.spines[['top', 'right']].set_visible(False)
    save_plot(fig, 'figure_features', 'paper_outputs')
convergence('best_fitness', 'Best fitness', 'figure_convergence', 'paper_outputs')
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for column, kernel in enumerate(KERNELS):
    draw_cm(axes[0, column], baselines[kernel]['test_tuned'], f'{kernel.upper()} baseline tuned')
    representative = representatives[kernel]
    draw_cm(axes[1, column], representative['test_tuned'], f'{kernel.upper()} GWO {representative["id"]}')
save_plot(fig, 'figure_kernel_confusions', 'paper_outputs')
def make_comparison(selected_by_kernel):
    rows = []
    for kernel in KERNELS:
        baseline = baselines[kernel]
        selected = selected_by_kernel[kernel]
        base_cv = final_table.loc[(final_table['ID'] == baseline['id']) &
                                 (final_table['Mode'] == 'tuned')].iloc[0]
        gwo_cv = final_table.loc[(final_table['ID'] == selected['id']) &
                                (final_table['Mode'] == 'tuned')].iloc[0]
        rows.append({'ratio': f'{round((1-TEST_SIZE)*100)}/{round(TEST_SIZE*100)}',
            'seed': split_seed, 'kernel': kernel, 'selected_scenario': selected['id'],
            'train_samples': len(y_train), 'test_samples': len(y_test),
            'features': selected['n_features'],
            'CV Train Acc Baseline (%)': base_cv['CV Train Acc (%)'],
            'CV Train Acc GWO (%)': gwo_cv['CV Train Acc (%)'],
            'CV Train F1 Baseline (%)': base_cv['CV Train F1 (%)'],
            'CV Train F1 GWO (%)': gwo_cv['CV Train F1 (%)'],
            'Test F1 Baseline (%)': base_cv['Test F1 (%)'],
            'Test F1 GWO (%)': gwo_cv['Test F1 (%)'],
            **{f'baseline_{m}': baseline['test_tuned'][m]*100 for m in ('Acc', 'BA')},
            **{f'gwo_{m}': selected['test_tuned'][m]*100 for m in ('Acc', 'BA')},
            **{f'delta_{m}_pp': (selected['test_tuned'][m]-baseline['test_tuned'][m])*100
               for m in ('Acc', 'BA')}})
    return pd.DataFrame(rows)

comparison = make_comparison(representatives)
_ = table('Baseline vs GWO terpilih dari CV train (metrik CV train dan TEST)',
          comparison, 'paired_comparison')

# Eksplorasi saja: accuracy TEST tertinggi, lalu fitur lebih sedikit dan ID terkecil.
# Tidak mengubah representatives, winner, parameter, maupun selection_lock.
test_representatives = {
    kernel: min((r for r in scenarios if r['kernel'] == kernel),
                key=lambda r: (-r['test_tuned']['Acc'], r['n_features'], int(r['id'][1:])))
    for kernel in KERNELS
}
comparison_test = make_comparison(test_representatives)
print('Eksploratif: dipilih memakai accuracy TEST; bukan estimasi evaluasi independen.')
_ = table('Baseline vs GWO terpilih dari TEST (eksploratif)',
          comparison_test, 'paired_comparison_test_exploratory')
print(f'\nSelesai. Seluruh tabel, rincian fold, model, dan plot: {RUN_DIR}')
