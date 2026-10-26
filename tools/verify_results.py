"""Check the single archived paper run without rerunning the optimizer."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/paper_run'
REPORTS = RUN / 'reports'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, help='Optionally check the original input and training variance ranking.')
args = parser.parse_args()

final = pd.read_csv(REPORTS / 'final_cv_train_vs_test.csv')
folds = pd.read_csv(REPORTS / 'final_cv_train_folds.csv')
recap = pd.read_csv(REPORTS / 'rekapitulasi.csv')
manifest = pd.read_csv(RUN / 'splits/sample_split.csv')
fold_manifest = pd.read_csv(RUN / 'splits/cv_sample_folds.csv')
lock = json.loads((REPORTS / 'selection_lock.json').read_text())
assert len(manifest) == 121 and manifest.sample_id.is_unique
assert manifest.groupby(['split', 'label']).size().to_dict() == {('test',0):29,('test',1):8,('train',0):65,('train',1):19}
assert len(fold_manifest) == 84 and set(fold_manifest.validation_fold) == set(range(1,11))
assert set(fold_manifest.sample_id) == set(manifest.loc[manifest.split=='train','sample_id'])
metric_fns = {'Acc':accuracy_score,'BA':balanced_accuracy_score,'Prec':precision_score,'Rec':recall_score,'F1':f1_score}
rows_checked = 0
for row in final.itertuples(index=False, name=None):
    r = dict(zip(final.columns, row))
    if r['ID'] != 'Dummy':
        pred = pd.read_csv(REPORTS / f"predictions_{r['ID']}_{r['Mode']}.csv")
        expected = manifest.loc[manifest.split=='test'].set_index('sample_id').label
        assert len(pred)==37 and pred.sample_id.is_unique
        assert set(pred.sample_id)==set(expected.index)
        assert np.array_equal(pred.actual.to_numpy(),expected.loc[pred.sample_id].to_numpy())
        for metric,fn in metric_fns.items():
            kwargs = {'zero_division':0} if metric in ('Prec','Rec','F1') else {}
            assert np.isclose(100*fn(pred.actual,pred.predicted,**kwargs),r[f'Test {metric} (%)'])
        rows_checked += 1
    selected = folds.loc[folds.Model==r['Model']]
    assert len(selected)==10
    for metric in metric_fns:
        assert np.isclose(100*selected[metric].mean(),r[f'CV Train {metric} (%)'])
        assert np.isclose(100*selected[metric].std(ddof=0),r[f'CV Train {metric} Std (pp)'])

scenarios = recap.loc[recap.ID.str.fullmatch(r'S[1-9]')].copy()
assert len(scenarios)==9
for r in scenarios.itertuples(index=False):
    probes=pd.read_csv(REPORTS/f'features_{r.ID}.csv')
    assert len(probes)==r.Fitur and probes.Probe.is_unique
    history=pd.read_csv(REPORTS/f'history_{r.ID}.csv')
    assert history.iteration.tolist()==list(range(1,41))
    assert np.isfinite(history.best_fitness).all()
    assert (history.best_fitness.diff().dropna()<=1e-12).all()
ranked=scenarios.assign(scenario_number=scenarios.ID.str[1:].astype(int)).sort_values(
    ['CV Joint Tuned (%)','Fitur','scenario_number'],ascending=[False,True,True])
assert ranked.iloc[0].ID==lock['winner']=='S9'
assert lock['features']==116 and lock['representatives']=={'linear':'S3','rbf':'S6','poly':'S9'}
for kernel,identifier in lock['representatives'].items():
    assert ranked.loc[ranked.Kernel.str.lower()==kernel].iloc[0].ID==identifier
assert lock['optimizer_iterations']==40

if args.data:
    path=args.data.expanduser().resolve()
    provenance=json.loads((RUN/'provenance.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==provenance['input_sha256'], 'Input checksum differs from the study CSV.'
    df=pd.read_csv(path)
    ids=df.iloc[:,0].str.strip()
    y=df.recurrence.to_numpy()
    assert len(df)==121 and np.isfinite(df.iloc[:,1:].drop(columns=['recurrence','stage']).to_numpy(dtype=float)).all()
    train,test=train_test_split(np.arange(len(df)),test_size=.30,stratify=y,random_state=42)
    assert set(ids.iloc[test])==set(manifest.loc[manifest.split=='test','sample_id'])
    values=df.drop(columns=[df.columns[0],'recurrence','stage']).astype(np.float32)
    assert values.shape[1]==2000
    ranking=values.iloc[train].astype(np.float64).var(axis=0,ddof=0).sort_values(ascending=False,kind='stable')
    saved=pd.read_csv(REPORTS/'train_variance_ranking.csv')
    assert ranking.index.tolist()==saved.Probe.tolist()
    assert np.allclose(ranking.to_numpy(),saved.train_variance.to_numpy())
    assert saved.selected.sum()==500
    print('Input checksum, split, and variance ranking verified.')

print(f'Verified {rows_checked} SVM test rows, 25 internal-CV summaries, nine feature subsets and histories, and the S9 selection lock.')
