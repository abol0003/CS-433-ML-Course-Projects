# Copilot Instructions for Project 1

## Project Overview
This is a machine learning project for CS-433, focused on training and evaluating models using provided datasets. The main workflow involves data preprocessing, model implementation, and running experiments via `run.py`.

## Key Files & Structure
- `run.py` / `run2.py`: Main entry points for running experiments and generating submissions.
- `implementations.py`: Contains all required ML functions with specific signatures. This is the core logic file.
- `helpers.py`: Utility functions for data handling and preprocessing.
- `data/`: Contains training and test datasets (`x_train.csv`, `y_train.csv`, `x_test.csv`).
- `data_saving/`: Stores intermediate results, model weights, and parameters.
- `grading_tests/`: Contains public test scripts and environment setup (`environment.yml`).

## Developer Workflows
- **Testing:**
  - Use the public tests in `grading_tests/test_project1_public.py`.
  - Run tests with: `pytest --github_link <GITHUB-REPO-URL> .` (or use a local directory for faster iteration).
  - Do NOT copy test files into your main repo; always run from the course repo.
- **Environment:**
  - Create grading environment: `conda env create --file=grading_tests/environment.yml --name=project1-grading`.
  - Activate: `conda activate project1-grading`.
- **Formatting:**
  - Use `black` for code formatting: `black <SOURCE-DIRECTORY>`.

## Project-Specific Patterns
- All main ML functions must be implemented in `implementations.py` with the requested signatures.
- Submission files should be generated as CSVs (see `submission_best.csv` and `data/sample-submission.csv`).
- Intermediate results and model parameters are saved in `data_saving/` as `.npz` or `.npy` files.
- Avoid hardcoding paths; use relative paths for portability.

## Integration Points
- No external web APIs; all data is local.
- Use only packages listed in `grading_tests/environment.yml`.
- Results and plots (e.g., confusion matrix, ROC curve) are saved in `picture/`.

## Conventions
- Keep all required function signatures in `implementations.py`.
- Main script should be `run.py` (or `run.ipynb` if using notebooks).
- Do not modify test files or grading scripts.
- Check for updates to public tests before final submission.

## Example: Running Tests Locally
```sh
conda env create --file=grading_tests/environment.yml --name=project1-grading
conda activate project1-grading
pytest --github_link <GITHUB-REPO-URL> grading_tests/
```

---
For questions about grading or environment setup, refer to `grading_tests/INSTRUCTIONS.md`.
