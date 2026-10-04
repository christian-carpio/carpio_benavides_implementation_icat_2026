# Effects of Preprocessing Schemes on Classifier Performance under Varying Percentages of Missing Values

Source code, data and results of the paper by **Christian R. Carpio** and **Luis E. Benavides** (Universidad de Especialidades Espíritu Santo, Ecuador), presented at ICAT 2026 and published in *Communications in Computer and Information Science* (Springer).

The study evaluates 27 preprocessing schemes (3 encoders × 3 imputers × 3 scalers) under three levels of missing values generated completely at random (MCAR) in the numerical variables, with three classifiers and stratified five-fold cross-validation. The effects of each preprocessing factor and their interactions are assessed with a three-way factorial ANOVA on the fold-level MCC.

## Repository structure

| Path | Content |
|---|---|
| `carpio_benavides_implementation.ipynb` | Complete pipeline: MCAR generation, preprocessing, training, evaluation, ANOVA, tables and charts |
| `heart.csv` | Original dataset (Heart Failure Prediction, see [Data](#data)) |
| `missing_values_datasets/` | Incomplete datasets with 10%, 20% and 30% MCAR (`heart_10.csv`, `heart_20.csv`, `heart_30.csv`) |
| `preprocessed_datasets/` | Training and test partitions after preprocessing, for every scheme, classifier and fold (2,430 files named `heart_{pct}_{encoding}_{imputation}_{scaler}_{model}_fold{r}_{train\|test}.csv`) |
| `results/fold_scores_all.csv` | Fold-level results: F1-Score and MCC of the 1,215 training runs (2,430 values) |
| `results/results_all.csv`, `results/results_heart_{10,20,30}.csv` | Mean and standard deviation over the five folds for each of the 243 models |
| `results/anova/` | ANOVA tables (Type II) for each MCAR level |
| `results/tables/` | Consolidated ANOVA table, main-effect and interaction tables, and rankings of all combinations |
| `results/graphs/` | MCC and F1-Score heatmaps |
| `requirements.txt` | Exact package versions |

## Experimental setup

| Component | Configuration |
|---|---|
| Random seed | 2025 |
| Missing values | MCAR applied only to the six numerical variables (918 × 6 = 5,508 cells). A single random permutation of the cells (NumPy `default_rng`, seed 2025) is drawn and its first 550, 1,101 and 1,652 cells are removed for 10%, 20% and 30%, so the masks are nested. Categorical variables and the target remain complete. |
| Encoding | Ordinal (`OrdinalEncoder`, unknown categories → −1), One-Hot (`OneHotEncoder`, unknown categories ignored), Target (smoothed mean with *m* = 10, unseen categories → global mean) |
| Imputation | Mean (`SimpleImputer`), k-NN (`KNNImputer`, *k* = 5, uniform weights), MICE (`IterativeImputer` with `BayesianRidge`, 25 iterations, tolerance 10⁻³, mean initialization, ascending order, seed 2025) |
| Scaling | Z-Score (`StandardScaler`), Min-Max (`MinMaxScaler`), Robust (`RobustScaler`) |
| Pipeline | `ColumnTransformer` with two parallel branches: numerical variables are imputed and then scaled; categorical variables are encoded and are not scaled |
| Classifiers | Gaussian Naive Bayes, k-NN (*k* = 5), SVM (RBF kernel, *C* = 1.0) |
| Validation | `StratifiedKFold` (5 folds, shuffled, seed 2025). The same folds are reused for every scheme and classifier, and every preprocessing step is fitted only on the training partition of each fold |
| Metrics | F1-Score and Matthews correlation coefficient (MCC) |
| Statistical analysis | Three-way factorial ANOVA (Type II, `statsmodels`) per MCAR level, on the 405 fold-level MCC values (27 schemes × 3 classifiers × 5 folds) |

## How to run

1. Install Python 3.13 (the published results were obtained with Python 3.13.8).
2. Create a virtual environment and install the dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Open `carpio_benavides_implementation.ipynb` and run all cells. The full experiment takes a few minutes on a laptop.

Running the notebook regenerates `missing_values_datasets/`, `preprocessed_datasets/` and `results/`, overwriting the existing files. To keep the published results, run it on a copy of the repository.

## Reproducibility note

The files in `results/` are the ones reported in the paper. They were obtained on macOS (Apple Silicon) with the package versions listed in `requirements.txt`.

The incomplete datasets and the cross-validation folds are reproduced exactly on any platform, as are the data preprocessed with mean imputation (MICE matches up to floating-point rounding). k-NN imputation, however, can differ slightly between platforms: when several donors are at the same distance from a record, the numerical libraries of each system may break the tie differently. This produces small differences in some fold-level metrics, but does not change the conclusions of the study.

## Data

The dataset is the *Heart Failure Prediction Dataset* published by fedesoriano on Kaggle (2021): <https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction>. It combines five heart-disease cohorts from the UCI Machine Learning Repository (Cleveland, Hungarian, Switzerland, Long Beach VA and Statlog) and contains 918 records after the removal of duplicates. Please refer to the dataset page for its license and terms of use.

## Citation

If you use this code or these results, please cite:

```bibtex
@inproceedings{carpio2026preprocessing,
  author    = {Carpio, Christian R. and Benavides, Luis E.},
  title     = {Effects of Preprocessing Schemes on Classifier Performance under Varying Percentages of Missing Values},
  booktitle = {ICAT 2026},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}
```

## License

The source code is released under the MIT License (see [LICENSE](LICENSE)). The dataset is subject to the terms of its original source.
