# Submitting through a Kaggle notebook

`pokemon-fds-final.ipynb` is a self-contained Kaggle notebook. It embeds the source of the modules
it needs, trains the final model on the competition data and writes `/kaggle/working/submission.csv`.

| File | Purpose |
|---|---|
| `build_kaggle_notebook.py` | Regenerates the notebook from `src/` and writes `kernel-metadata.json` |
| `pokemon-fds-final.ipynb` | The notebook (built, not executed) |
| `kernel-metadata.json` | Notebook settings for the Kaggle CLI: owner, title, competition data, no GPU, no internet |

Rebuild the notebook every time `src/` changes:

```bash
python kaggle/build_kaggle_notebook.py --user <kaggle-username>
```

Expected runtime on Kaggle's CPU (4 cores): about 30–40 minutes.

## Option 1: Kaggle website only

1. Open the competition, then **Code** → **New Notebook**.
2. **File** → **Import Notebook**, and choose `kaggle/pokemon-fds-final.ipynb`.
3. In the right panel:
   - **Input:** the competition data must be attached (add it with **Add Input** if missing);
   - **Accelerator:** None;
   - **Internet:** off.
4. **Save Version** → **Save & Run All (Commit)**.
5. When the run finishes, open the notebook viewer and, under **Output**, click **Submit to
   Competition**.

## Option 2: Kaggle CLI

### One-time setup

```bash
pip install kaggle            # CLI >= 2.0
kaggle auth login             # opens the browser and links the CLI to your Kaggle account
```

Instead of `auth login` you can use a legacy API token:
1. Go to kaggle.com → **Settings** → **API** → **Create New Token**. This downloads `kaggle.json`.
2. Move it to `~/.kaggle/kaggle.json`.
3. Run `chmod 600 ~/.kaggle/kaggle.json`.

Then join the competition **on the website** (open its page and accept the rules), otherwise every
command returns 403. Check that it works:

```bash
kaggle competitions files fds-pokemon-battles-prediction-2026
```

### Run the notebook on Kaggle and submit it

```bash
# 1. set your Kaggle username as the notebook owner (once)
python kaggle/build_kaggle_notebook.py --user <kaggle-username>

# 2. upload the notebook and run it on Kaggle (creates version 1, then 2, 3, ... on every push)
kaggle kernels push -p kaggle

# 3. follow the run until the status is "complete" (about 30-40 minutes)
kaggle kernels status <kaggle-username>/pokemon-fds-final

# 4. optional: download the outputs (submission.csv and the run log)
kaggle kernels output <kaggle-username>/pokemon-fds-final -p kaggle/output

# 5. submit the notebook's submission.csv (use the version number of the run from step 2)
kaggle competitions submit fds-pokemon-battles-prediction-2026 \
    -k <kaggle-username>/pokemon-fds-final -v 1 -f submission.csv \
    -m "final 10-model stack" --wait

# 6. score and history of your submissions
kaggle competitions submissions fds-pokemon-battles-prediction-2026
```

If the competition also accepts uploaded files, you can skip the notebook and submit the file
produced locally by `src/final_model.py`:

```bash
kaggle competitions submit fds-pokemon-battles-prediction-2026 \
    -f outputs/submission_final.csv -m "final 10-model stack (local run)"
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `403 Forbidden` | Accept the competition rules on the website with the same account |
| `401 Unauthorized` | Run `kaggle auth login --force` again, or check `~/.kaggle/kaggle.json` |
| `kernels push` fails on the title | The title must match the slug (`Pokemon FDS Final` → `pokemon-fds-final`); do not edit one without the other |
| The run stops with "data not found" | The competition data is not attached: check `competition_sources` in `kernel-metadata.json` |
| Run status `error` | `kaggle kernels output ... -p kaggle/output` downloads the log; the failing step is printed there |
