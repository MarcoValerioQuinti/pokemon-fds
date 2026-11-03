# Kaggle notebook

`pokemon-fds-final.ipynb` is the notebook submitted for the competition:
- it clones this repository at a fixed commit (no project code is pasted into the notebook);
- it runs the pipeline on the competition data;
- it writes the 3 selected final submissions to `/kaggle/working`, each preceded by a markdown
  cell that explains the approach.

| File | Purpose |
|---|---|
| `build_kaggle_notebook.py` | Generates the notebook and `kernel-metadata.json` |
| `pokemon-fds-final.ipynb` | The notebook (built, not executed) |
| `kernel-metadata.json` | Settings for the Kaggle CLI: owner, title, competition data, CPU, internet on (to clone) |

The notebook pins a commit, so rebuild it after pushing any change to `src/`:

```bash
git push
python kaggle/build_kaggle_notebook.py --user <kaggle-username>   # pins the current HEAD
```

The output files are:
- `submission_1_stack10.csv`
- `submission_2_stack22.csv`
- `submission_3_stack10_mean.csv`

The run takes about 1 hour on Kaggle's CPU (4 cores).

## From the Kaggle website

1. Open the competition, then **Code** → **New Notebook** → **File** → **Import Notebook**, and
   choose `kaggle/pokemon-fds-final.ipynb`.
2. In the right panel:
   - **Input:** the competition data must be attached;
   - **Accelerator:** None;
   - **Internet:** on.
3. **Save Version** → **Save & Run All (Commit)**.
4. When the run finishes, open the notebook viewer and, under **Output**, submit each of the three
   CSV files with **Submit to Competition**.
5. Before the deadline, select the three submissions on the competition's **Submissions** page.

After the competition ends, make the notebook public.

## From the Kaggle CLI

One-time setup:

```bash
pip install kaggle            # CLI >= 2.0
kaggle auth login             # links the CLI to your Kaggle account in the browser
```

Then accept the competition rules on the website (otherwise every command returns 403).

```bash
kaggle kernels push -p kaggle                              # uploads and RUNS the notebook (new version)
kaggle kernels status <user>/pokemon-fds-final             # wait for "complete"
kaggle kernels output <user>/pokemon-fds-final -p kaggle/output
kaggle competitions submit fds-pokemon-battles-prediction-2026 \
    -k <user>/pokemon-fds-final -v <version> -f submission_1_stack10.csv -m "stack10"
kaggle competitions submissions fds-pokemon-battles-prediction-2026
```

The competition allows 3 submissions per day.

The CLI cannot cancel a running notebook: cancel it from the website. `kaggle kernels delete`
removes the whole notebook.
