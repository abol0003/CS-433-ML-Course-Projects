# Copilot Instructions for Project 1

Purpose: Help AI agents work productively in this ML repo (CS‑433 Project 1) by documenting the actual architecture, workflows, and conventions in code.

## Big picture
- Task: binary classification with linear models (GD/SGD MSE, Least Squares, Ridge, Logistic, Regularized Logistic) trained on local CSVs.
- Pipeline (in `run.py`):
  1) Load CSVs via `helpers.load_csv_data` (ID in first column; labels are −1/+1)
  2) `preprocessing.preprocess`: mean-impute (by TRAIN), drop constant/NA-only cols, light one‑hot (capped), standardize (by TRAIN), prepend bias 1.
  3) Tune λ, γ with stratified k‑fold CV; pick global threshold maximizing F1 from concatenated fold probabilities.
  4) Train final reg-logistic on full train; save weights, build submission, save plots.

## Key files
- `run.py`: Primary entrypoint; uses only local numpy/matplotlib. Controls caching to `data_saving/` and outputs to `picture/`.
- `run2.py`: Alternate runner with similar pipeline and knobs (Adam/early stop flags) — use only if compatible with current `implementations.py`.
- `implementations.py`: Required functions and signatures used by grading tests: `mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression(y, tx, lambda_)`, `logistic_regression`, `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, ...)`, plus helpers like `sigmoid`, `logistic_loss`, `logistic_gradient`.
- `cv_utils.py`: Stratified splits and CV loop (`cv_train_and_eval`) selecting F1‑optimal threshold.
- `preprocessing.py`: Imputation, feature filtering, capped one‑hot (see constants), standardization, bias column.
- `metrics.py`, `plots.py`, `helpers.py`: Metrics/label conversion, plotting, CSV IO.

## Conventions and gotchas
- Labels: helpers returns y in {−1,+1}; convert to {0,1} early via `metrics.to_01_labels`. Keep types consistent end‑to‑end.
- Preprocessing caps (from `config.py`): `LOW_CARD_MAX_UNIQUE`, `ONEHOT_PER_FEAT_MAX`, `MAX_ADDED_ONEHOT`. Do not exceed caps; they keep dims small.
- Paths: use `config.py` constants (`DATA_DIR`, `SAVE_*`, `PICT_DIR`). All IO is relative; avoid hard‑coded absolute paths.
- Caching: `data_saving/preprocessed_data.npz`, `best_params.npz`, `final_weights.npy` are reused based on `config.DO_*` toggles.
- Windows: multiprocessing uses `spawn`; `run.py` sets `matplotlib.use("Agg")` for headless plot saving.
- API stability: Do not change function names/signatures in `implementations.py` (grading depends on them). If you add optional kwargs, keep backward‑compat.
- Packages: stick to `grading_tests/environment.yml` (numpy/matplotlib/etc.); no sklearn for training logic.

## Workflows
- Environment
  - Create: `conda env create --file=grading_tests/environment.yml --name=project1-grading`
  - Activate: `conda activate project1-grading`
- Run public tests (from the course repo):
  - `pytest --github_link <GITHUB-REPO-URL> grading_tests/`
- Run pipeline:
  - Adjust toggles in `config.py` (`DO_PREPROCESS`, `DO_TUNE`, `DO_SUBMISSION`, seeds, iter budgets).
  - Place CSVs under `data/dataset/` (with ID in col 0).
  - `python run.py` produces `submission_best.csv`, caches to `data_saving/`, and figures under `picture/`.
- Formatting: use `black` on source files.

## Examples (repo‑specific)
- CV thresholding: see `cv_utils.best_threshold_by_f1` and its use in `cv_train_and_eval` (global threshold over concatenated folds).
- Preprocess shape logging and caps: see `preprocessing.preprocess` prints and one‑hot caps from `config.py`.

Questions or unclear bits? Ping to refine these instructions (e.g., if you plan to extend `implementations.reg_logistic_regression` with Adam/early‑stop while keeping signatures stable).
