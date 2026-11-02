# Kaggle: original pipeline

The notebook now uses `git clone` and a pinned Git commit instead of embedding Python modules.
It runs the original `simulate.py`, `recursive.py` and `final_model.py`, preserving their model
parameters, seeds and features. The weaker data-only alternative and its CSVs have been removed.

Use CPU and enable Internet for cloning/installing dependencies. Attach the official
`train.jsonl`, `test.jsonl`, `type_chart_gen2.csv`, and `sample_submission.csv` together.
Discovery requires exactly one matching folder, otherwise set `DATA_DIR`. Use a fresh output
folder to avoid stale original caches. If Internet is prohibited, clarify the cloning rule with
organizers. No unverified dataset slug is configured.

After reviewing and publishing the authorized repository revision, set its full SHA in the notebook
or rebuild it with `python kaggle/build_kaggle_notebook.py --commit FULL_40_CHARACTER_SHA`.
The revision must include `requirements-final.txt`. No push or publication has been performed.
Original ownership metadata is preserved; use only an authorized team account.

The output is `/kaggle/working/submission_final.csv`. Local reproduction gave 0.69958 repeated
meta-CV accuracy (SD 0.01009). The team supplied these existing public leaderboard scores:

| Entry | Owner | Public score |
|---|---|---:|
| submission_final.csv | Marco Valerio Quinti | 0.70133 |
| submission_stacking_mc_rec.csv | Marco Valerio Quinti | 0.70066 |
| notebook Version 1 | Luca Martone | 0.69466 |

These scores were supplied by the user, not independently retrieved, and are distinct from CV.
The revised cloning notebook has not been run on Kaggle. Local regenerated CSV byte identity
with Marco's uploaded file has not been established.

The notebook currently reproduces only the original final configuration. Recover the exact
configuration for Marco's broader stack and the code/configuration for Luca's Version 1 before
claiming reproducibility of all three entries. No arbitrary replacement models or duplicate
submissions are generated. No further tests/training were run after the user's stop request.

Check unresolved mechanics provenance and team-sharing questions in `COMPLIANCE.md`. Uploading
and selecting entries remain manual: at most three per day; select three before November 5,
2026, 23:59, confirming official timezone. Make the notebook public after the competition;
finalize the deferred report and deliver it with the notebook by November 12, 2026.
