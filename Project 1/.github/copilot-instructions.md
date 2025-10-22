# Copilot Instructions for Project 1

Purpose: Enable AI agents to be productive in this ML repo (CS-433 Project 1) by capturing the actual architecture, workflows, and conventions used in code.
# Copilot Instructions for Project 1

Purpose: Make AI agents effective in this ML repo (CS‑433 Project 1) by capturing the actual architecture, workflows, and conventions used in code.

## Big picture
- Task: Binary classification on tabular CSVs using linear models (GD/SGD MSE, Least Squares, Ridge, Logistic, Regularized Logistic) implemented with NumPy only.
- Pipeline (see `run.py`):
  1) Load CSVs via `helpers.load_csv_data(DATA_DIR)` → returns `(x_train, x_test, y_train_pm1, train_ids, test_ids)`; labels are −1/+1.
  2) Preprocess with `preprocessing.preprocess2`:
     - remove low‑validity features, smart impute, variance threshold, drop highly correlated features,
       light one‑hot (capped by config), standardize on TRAIN stats, then prepend bias 1.
     - Saves to `config.PREPROC2_DATA_PATH` (npz with X_train, X_test, y_train, ids).
  3) Hyperparameter tuning: sample λ and γ from log‑uniform ranges and run stratified K‑fold CV (`cv_utils.cv_train_and_eval`).
     - Select a single global probability threshold maximizing F1 over concatenated validation scores.
  4) Train final regularized logistic on full train with best (λ, γ), save weights, build submission.

## Key files
- `run.py`: Entrypoint. Provides helpers for preprocess/tune/train/submit. Uses `matplotlib.use("Agg")` and multiprocessing spawn (Windows‑friendly).
- `config.py`: All paths and knobs (DATA_DIR, PICT_DIR, SAVE_BEST, SAVE_WEIGHTS, PREPROC2_DATA_PATH; encoding caps and search ranges).
- `implementations.py`: Graded API. Keep function names/signatures: `mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression(y, tx, lambda_)`, `logistic_regression`, `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, display=False, adam=True, schedule=None, early_stopping=False, patience=10, tol=1e-6, verbose=False, callback=None)`, `sigmoid`, `logistic_loss`, `logistic_gradient`.
  - Note: `reg_logistic_regression` supports Adam, optional LR schedules, early stopping, and a callback; defaults preserve backward compatibility. It returns the UNPENALIZED loss (the L2 penalty is used only during training). Keep the public signature stable.
- `cv_utils.py`: `stratified_kfold_indices`, `best_threshold_by_f1`, and `cv_train_and_eval` (returns λ, γ, best_thr, and averaged metrics).
- `preprocessing.py`: Implements `preprocess2` steps above; capped one‑hot per `config` with shape/cap logs.
- `metrics.py`, `helpers.py`, `plots.py`: Label conversion and metrics, CSV IO (`create_csv_submission`), plotting curves and confusion matrix.

## Conventions and gotchas
- Labels: Convert {−1,+1} → {0,1} early via `metrics.to_01_labels`; all logistic code expects {0,1}. Convert back with `metrics.to_pm1_labels` for submissions.
- One‑hot caps from `config.py`: `LOW_CARD_MAX_UNIQUE`, `ONEHOT_PER_FEAT_MAX`, `MAX_ADDED_ONEHOT` — do not exceed; keeps dims bounded.
- Standardization: compute mean/std on TRAIN only; apply to TEST; bias column is added after standardization as the first column.
- Paths and caching: prefer `config.PREPROC2_DATA_PATH`, `config.SAVE_BEST`, `config.SAVE_WEIGHTS`, and `PICT_DIR` images. Avoid hard‑coded paths.
- Toggles: both legacy `DO_PREPROCESS/DO_TUNE/DO_SUBMISSION` and newer `PREPROCESSING/HYPERPARAM_TUNING/SUBMISSION` appear in configs/code. Don’t mix them in the same path. Prefer the newer names and forward/alias the legacy ones to a single source of truth.
  - Heads‑up: If code references undefined toggles like `SAVE_PREPROCESSED`, replace with `PREPROC2_DATA_PATH` consistently.
- API stability: keep `implementations.py` signatures unchanged. Ensure internal calls use matching kwargs (e.g., `logistic_gradient(..., lambda_=...)`). No sklearn.
- Windows: multiprocessing uses `mp.get_context("spawn")`; plotting is headless via `Agg`.

Additional notes on regularization
- For evaluation and reporting, use unpenalized logistic loss (`lambda_=0`) and compute probabilities with `sigmoid(X @ w)`.
- L2 regularization typically should not penalize the bias term. Ensure consistency between `reg_logistic_regression` and `logistic_loss` regarding bias handling.

## Workflows
- Environment
  - Create: `conda env create --file=grading_tests/environment.yml --name=project1-grading`
  - Activate: `conda activate project1-grading`
- Tests: run locally with `pytest grading_tests/`.
- Run pipeline
  - Place CSVs under `data/dataset/` (ID in col 0).
  - Adjust ranges/budgets and caps in `config.py` (GAMMA/LAMBDA ranges, N_TRIALS, *_MAX_ITERS).
  - Execute `python run.py`. The `main()` wires `preprocess2 → tune_hyperparameter → train_final_model → make_submission` depending on toggles.
  - Outputs: `submission_best.csv`, caches in `data_saving/`, figures in `picture/`.

## Examples from code
- CV thresholding: `cv_utils.best_threshold_by_f1` on concatenated fold probabilities; reused threshold to score each fold and average metrics.
- Preprocess logs: `preprocessing.one_hot_encoding` prints counts and caps; `variance_treshold` and correlation pruning report kept/removed cols.

## run.py cleanup checklist (please apply)
- Unify toggles: prefer `PREPROCESSING`, `HYPERPARAM_TUNING`, and `DO_SUBMISSION` (or `SUBMISSION`) consistently. Remove duplicated legacy flags, or alias old names to the new ones in `config.py` and reference only one set inside `run.py`.
- Fix hyperparameter sampling: replace hardcoded single values for `lambda_samples` and `gamma_samples` with proper log‑uniform sampling using `sample_loguniform(...)`, sized to `N_TRIALS`. Ensure `cv_utils.cv_train_and_eval` receives the sampled grids.
- Multiprocessing: keep `mp.get_context("spawn").Pool(...)` for Windows. Compute `nproc` conservatively.
- Scheduling: centralize LR schedule selection from `config.SCHEDULE_DEFAULT` ("none" | "cosine" | "exponential"), pass a callable to `reg_logistic_regression`, and avoid duplicating schedule logic.
- Naming consistency: use `y_tr_01` (or `y01`) consistently; avoid mixing `ytr_01` and other variants. Keep feature matrices named `X_tr`, `X_te`.
- Remove dead/legacy code: delete the commented legacy `main()` block and unused imports. Keep a single `main()`.
- I/O paths: always use `config.PREPROC2_DATA_PATH`, `config.SAVE_BEST`, and `config.SAVE_WEIGHTS`. Do not introduce undefined toggles or ad‑hoc paths.
- Reproducibility: set RNG seed once (NumPy + any optimizer-seeded RNG as needed). Avoid reseeding inside loops unless required.
- Keep helpers focused: ensure `tune_hyperparameter`, `train_final_model`, and `make_submission` have minimal side effects beyond intended saves.

Questions or unclear bits? Open an issue/PR to clarify, especially around the config toggle naming and the legacy `SAVE_PREPROCESSED` vs `PREPROC2_DATA_PATH`.
