# Study input

The experiment expects `GSE21510_recurrence_top2000.csv`, the processed input used in the study. This repository publishes the experiment code and saved reports, but does not redistribute the supplied expression CSV. Use your copy of that input to rerun the optimization.

Default location:

```text
data/GSE21510_recurrence_top2000.csv
```

Alternatively, pass `--data /path/to/file.csv` to the Python script.

## Expected format

- 121 rows, one per sample.
- First column: unique `GSM...` sample accession; the study CSV names it `Unnamed: 0`.
- `recurrence`: binary target, `0` = no recurrence (94 rows), `1` = recurrence (27 rows).
- `stage`: metadata column required by the exploration code; excluded from predictors.
- 2,000 numeric expression-probe columns, finite values, no duplicate column names.
- Preserve the original row and column order to reproduce the saved split and stable variance ranking.

Expected SHA-256 of the exact study input bytes:

```text
3c7793dab9ad9e7b4f233e9f10db85e8660657d027c2f10ba15819230aa99d0b
```

The experiment ranks variance on the 84 training samples (`ddof=0`) and retains 500 probes for baseline SVMs and GWO selection. GSM identifiers, recurrence labels, and stage are excluded from predictors.

## Provenance

The expression data are derived from [GSE21510](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE21510), platform GPL570. The clinical derivation/follow-up definition of the supplied binary labels and the upstream normalization/filtering were not independently reconstructed. The processed CSV should not be presented as an official GEO recurrence-label export or as an independently reproduced preprocessing pipeline.

Saved reports, split assignments, and predictions use GEO sample accessions. No expression matrix or fitted model binaries are included in the published results folder.
