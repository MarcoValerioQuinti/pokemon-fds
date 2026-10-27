# Data

The competition data is not included in this repository: it is subject to the competition rules,
and `train.jsonl` (164 MB) exceeds GitHub's 100 MB file-size limit.

Download it from the
[competition page](https://www.kaggle.com/competitions/fds-pokemon-battles-prediction-2026/data),
or use the Kaggle CLI. The CLI needs an API token in `~/.kaggle/kaggle.json` and the competition
rules accepted on the website.

```bash
pip install kaggle
kaggle competitions download -c fds-pokemon-battles-prediction-2026 -p data
unzip data/fds-pokemon-battles-prediction-2026.zip -d data
```

Expected layout:

```
data/
  train.jsonl              10,000 battles with the label player_won
  test.jsonl               5,000 battles to predict
  type_chart_gen2.csv      Gen 2 type-effectiveness chart
  sample_submission.csv    submission format (battle_id, player_won)
```
