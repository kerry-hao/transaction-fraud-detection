# Setup, data access and GitHub

## You can read the project now

Open README, the notebook, methodology and reference audit without competition data.
The notebook has separate switches for reading data, training, holdout evaluation and final
submission generation. All are initially false.

## Manual download: easiest first run

1. Sign in to Kaggle and open the [competition data page](https://www.kaggle.com/competitions/ieee-fraud-detection/data).
2. Review and accept the competition access requirements if prompted.
3. Download and extract the five CSV files. Place them directly in `data/raw/`.
4. Install the project and run `python -m fraud_detection.cli check`.

You do not need the Kaggle API to use any modelling code. Downloading in a browser and downloading
through the CLI provide the same underlying competition files.

## Optional Kaggle CLI: convenient for repeated setups

Install the optional dependency and follow the current
[official authentication instructions](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md).
Keep all credentials outside the repository. After authenticating and accepting competition access:

```bash
python -m pip install -e ".[download]"
kaggle competitions download -c ieee-fraud-detection -p data/raw
```

Extract the downloaded archive, including any nested CSV archives, into `data/raw/`. This download
command is documentation only; no project script calls it. If access fails, first check your Kaggle
login and competition access on the website. Never paste credentials into notebooks or Git commits.

## Upload the project to GitHub

Suggested repository name: `ieee-cis-fraud-detection`.
Suggested description: `Transaction fraud modelling with chronological validation, LightGBM and review-capacity analysis. Learning project in progress.`

Unzip the package and use the contents of the `ieee-cis-fraud-detection` folder as the repository root.
Do not upload the ZIP as the only repository file. With GitHub's web interface, upload the source
folders and files; hidden files such as `.gitignore` and `.github/workflows/checks.yml` may need
separate attention. A local Git commit preserves these files more reliably.

For a new empty GitHub repository, run these commands inside the extracted project folder:

```bash
git init
git branch -M main
git add .
git status
git commit -m "Add IEEE-CIS fraud detection learning pipeline"
git remote add origin https://github.com/YOUR_USERNAME/ieee-cis-fraud-detection.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. Review `git status` before the commit; only
source, configuration and documentation should be staged. If a remote already contains commits,
clone it first, copy this package's contents into that clone, then commit and push normally.
Avoid force-pushing over existing work.

## First execution checklist

- Use Python 3.11/3.12 and install from the repository root.
- Confirm the five CSV names; training/EDA require only the two labelled files.
- Read `configs/default.toml` and keep the initial experiment definition fixed for the first run.
- Run `inspect`, then `train`, then `evaluate` with the printed run directory.
- Check saved logistic convergence warnings and compare prevalence with mean predicted probability.
- Update `reports/results.md` with measured aggregate results and their run ID.
- Generate a final submission only if you want hidden-test predictions.

## Current verification boundary

The delivered source has been syntax-checked and tested with small invented fixtures. No classifier
has been fitted and no competition CSV has been loaded. The LightGBM training path, full-data memory
use, convergence, timing and competition scores are not empirically verified yet. A successful
code-only check is not an end-to-end execution result.
