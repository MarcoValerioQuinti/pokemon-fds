# Official competition inputs

Obtain these four files from the intended competition page using your authorized account after
accepting its rules:

```text
train.jsonl
test.jsonl
type_chart_gen2.csv
sample_submission.csv
```

Place them directly in this directory, or pass their shared parent to
`POKEMON_DATA=/path/to/official/files python src/final_model.py`.
The same variable configures the simulation and recursive modules. On Kaggle the notebook discovers an input
folder containing all four files, or accepts an explicit `DATA_DIR` configuration.

No competition/dataset slug is assumed by the new workflow. Confirm the intended year and
competition before downloading or uploading anything. The task's 2026 rules and dates were not
independently verified through the public rules page during this audit.

Data and local symlinks are ignored by Git. The current workspace uses ignored symlinks to the
four user-provided files; these links are a local convenience and are not part of the deliverable.
Do not commit datasets, API tokens, or private raw prediction caches.
