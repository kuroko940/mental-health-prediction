# =============================================================================
# Projet Machine Learning M1 IDD -- Indicateurs d'Anxiete et de Depression
# Auteurs : Magomed Tsitsiev, Nouannapha Phichith
# =============================================================================

import argparse
import os
import pickle
import json
import numpy as np
import pandas as pd
from time import gmtime, strftime


from sklearn.model_selection import cross_val_score, GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =============================================================================
# Arguments
# =============================================================================
parser = argparse.ArgumentParser(
    prog="ML Project -- Anxiety & Depression Indicators",
    description=(
        "Projet ML M1 IDD : prediction des taux de depression "
        "aux Etats-Unis a partir de donnees socio-economiques (panel 2020-2024).\n"
        "Trois taches : regression, classification, clustering.\n"
        "Split par Etat (GroupShuffleSplit) pour eviter toute fuite de donnees."
    )
)

parser.add_argument("--dataset_path", type=str,
                    default="docs/Indicators_of_Anxiety_or_Depression.csv",
                    help="path to the main dataset (CSV)")
parser.add_argument("--socio_path", type=str,
                    default="docs/master_socioeconomic_data_2020_2024.csv",
                    help="path to the panel socioeconomic dataset 2020-2024 (CSV)")
parser.add_argument("--task", type=str, default="regression",
                    choices=["regression", "classification", "clustering"],
                    help="ML task: 'regression', 'classification' or 'clustering'")
parser.add_argument("--ml_method", type=str, default="knn",
                    help=(
                        "ML method to use.\n"
                        "  Regression    : 'linear_reg', 'ridge', 'sgd', 'knn', 'svr', 'random_forest', 'gradient_boosting'\n"
                        "  Classification: 'log_reg', 'gnb', 'knn', 'svm', 'random_forest', 'gradient_boosting'"
                    ))
parser.add_argument("--knn_n_neighbors", type=int, default=30,
                    help="number of neighbors for KNN (default: 30)")
parser.add_argument("--cv_nsplits", type=int, default=5,
                    help="number of folds for cross-validation (default: 5)")
parser.add_argument("--kmeans_k", type=int, default=3,
                    help="number of clusters for K-Means (default: 3)")
parser.add_argument("--threshold", type=int, default=22,
                    help="critical threshold (%%) for binary classification (default: 22)")
parser.add_argument("--save_dir", type=str, default="results",
                    help="root directory for saving model, logs and config")

args = parser.parse_args()

# =============================================================================
# Output directory (timestamped)
# =============================================================================
dir_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
out_dir = os.path.join(args.save_dir, f"{args.task}_{dir_name}")
os.makedirs(out_dir)

path_model  = os.path.join(out_dir, "model.pkl")
path_config = os.path.join(out_dir, "config.json")
path_logs   = os.path.join(out_dir, "logs.json")

with open(path_config, "w") as f:
    json.dump(vars(args), f, indent=2)

print(f"\n{'='*70}")
print(f"  Tache      : {args.task.upper()}")
print(f"  Methode    : {args.ml_method}")
print(f"  Resultats  : {out_dir}")
print(f"{'='*70}\n")

# =============================================================================
# Loading and cleaning the main dataset
# =============================================================================
print(">>> Chargement du dataset principal ...")
df = pd.read_csv(args.dataset_path)
print(f"    Shape brut : {df.shape}")

df['Time Period Start Date'] = pd.to_datetime(df['Time Period Start Date'])
df['Year']  = df['Time Period Start Date'].dt.year

# Keep only "By State" rows
df_by_state = df[df['Group'] == 'By State'].copy()
print(f"    Shape apres filtrage (By State) : {df_by_state.shape}")

# =============================================================================
# Merging with panel socioeconomic data (2020-2024)
# =============================================================================
print("\n>>> Fusion avec les donnees socio-economiques (panel 2020-2024)...")
df_socio = pd.read_csv(args.socio_path)
print(f"    Socio panel shape : {df_socio.shape}")

# Merge on State AND Year (panel merge) -- identique au notebook
df_merged = pd.merge(
    df_by_state, df_socio,
    on=['State', 'Year'],
    how='inner'
)
df_merged = df_merged.dropna(subset=['Value', 'median_household_income'])

# Keep only depression for tasks 1 & 2
df_depression = df_merged[
    df_merged['Indicator'] == 'Symptoms of Depressive Disorder'
].copy()

print(f"    Shape apres fusion  : {df_merged.shape}")
print(f"    Depression seule    : {df_depression.shape}")

SOCIO_COLS = [
    'median_household_income', 'poverty_rate',
    'pct_bachelors_or_higher', 'pct_uninsured', 'unemployment_rate'
]

# =============================================================================
# TASK 1 : SUPERVISED REGRESSION
# Predict depression symptom rate (Value) from socioeconomic profile
# Split by State (GroupShuffleSplit) to avoid data leakage
# =============================================================================
if args.task == "regression":
    print("\n" + "="*70)
    print("  TACHE 1 : REGRESSION SUPERVISEE")
    print("  (Depression uniquement, split par Etat)")
    print("="*70)

    X = df_depression[SOCIO_COLS].copy()
    y = df_depression['Value'].copy()
    groups = df_depression['State'].copy()

    print(f"\n  Features : {SOCIO_COLS}")
    print(f"  Echantillons : {X.shape[0]}  |  Features : {X.shape[1]}")

    # Group-based split: test states never seen during training
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    n_train_states = groups.iloc[train_idx].nunique()
    n_test_states = groups.iloc[test_idx].nunique()
    print(f"  Train : {X_train.shape[0]} ({n_train_states} Etats)")
    print(f"  Test  : {X_test.shape[0]} ({n_test_states} Etats, jamais vus)")

    # --- Build model (hyperparamètres alignés sur le notebook) ---
    if args.ml_method == "linear_reg":
        model = LinearRegression()
    elif args.ml_method == "ridge":
        model = Ridge(alpha=100)
    elif args.ml_method == "sgd":
        model = SGDRegressor(alpha=0.1, penalty='l1', max_iter=1000, random_state=42)
    elif args.ml_method == "knn":
        model = KNeighborsRegressor(n_neighbors=args.knn_n_neighbors,algorithm='brute') # k=30 tuné
    elif args.ml_method == "svr":
        model = SVR(kernel='rbf', C=1, gamma='scale')
    elif args.ml_method == "random_forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    elif args.ml_method == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        )
    else:
        raise ValueError(f"Unknown regression method: '{args.ml_method}'.")

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[SOCIO_COLS] = scaler.fit_transform(X_train[SOCIO_COLS])
    X_test_scaled[SOCIO_COLS] = scaler.transform(X_test[SOCIO_COLS])

    # --- Cross-validation (5 folds) avec GroupKFold pour éviter les fuites ---
    print(f"\n  Validation croisee ({args.cv_nsplits} folds) en cours...")
    gkf = GroupKFold(n_splits=args.cv_nsplits)
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train,
        groups=groups_train,
        cv=gkf, scoring='neg_mean_absolute_error'
    )
    cv_mae_mean = -cv_scores.mean()
    cv_mae_std  =  cv_scores.std()
    print(f"  CV MAE : {cv_mae_mean:.3f} +/- {cv_mae_std:.3f}")

    # --- Train and evaluate ---
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    test_mae  = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2   = r2_score(y_test, y_pred)

    print(f"\n  RESULTATS SUR LE TEST SET")
    print(f"  {'MAE':<10} : {test_mae:.3f}")
    print(f"  {'RMSE':<10} : {test_rmse:.3f}")
    print(f"  {'R2':<10} : {test_r2:.3f}")

    logs = {
        "task": "regression",
        "ml_method": args.ml_method,
        "indicator": "Symptoms of Depressive Disorder",
        "n_train_states": n_train_states,
        "n_test_states": n_test_states,
        "cv_mae_mean": round(cv_mae_mean, 4),
        "cv_mae_std":  round(cv_mae_std, 4),
        "test_mae":    round(test_mae, 4),
        "test_rmse":   round(test_rmse, 4),
        "test_r2":     round(test_r2, 4)
    }

# =============================================================================
# TASK 2 : SUPERVISED CLASSIFICATION (binary)
# Predict if depression rate >= threshold (default: 22%)
# Uses only socioeconomic features (no Year, no Indicator, no Value)
# Split by State (GroupShuffleSplit) to avoid data leakage
# =============================================================================
elif args.task == "classification":
    print("\n" + "="*70)
    print("  TACHE 2 : CLASSIFICATION BINAIRE")
    print(f"  (Depression >= {args.threshold}%, split par Etat)")
    print("="*70)

    df_clf = df_depression.copy()
    df_clf['is_critical'] = (df_clf['Value'] >= args.threshold).astype(int)

    X = df_clf[SOCIO_COLS].copy()
    y = df_clf['is_critical'].copy()
    groups = df_clf['State'].copy()

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"\n  Classe 0 (< {args.threshold}%) : {n_neg} ({100*n_neg/len(y):.1f}%)")
    print(f"  Classe 1 (>= {args.threshold}%) : {n_pos} ({100*n_pos/len(y):.1f}%)")
    print(f"\n  Features : {SOCIO_COLS}")
    print(f"  (Pas de Year, pas d'Indicator, pas de Value)")
    print(f"  Echantillons : {X.shape[0]}  |  Features : {X.shape[1]}")

    # Group-based split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    n_train_states = groups.iloc[train_idx].nunique()
    n_test_states = groups.iloc[test_idx].nunique()
    print(f"  Train : {X_train.shape[0]} ({n_train_states} Etats)")
    print(f"  Test  : {X_test.shape[0]} ({n_test_states} Etats, jamais vus)")

    # --- Build model (hyperparamètres alignés sur le notebook) ---
    if args.ml_method == "log_reg":
        model = LogisticRegression(C=1, max_iter=1000, random_state=42)
    elif args.ml_method == "gnb":
        model = GaussianNB()
    elif args.ml_method == "knn":
        model = KNeighborsClassifier(n_neighbors=20)
    elif args.ml_method == "svm":
        model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    elif args.ml_method == "random_forest":
        
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    elif args.ml_method == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        )
    else:
        raise ValueError(f"Unknown classification method: '{args.ml_method}'.")

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[SOCIO_COLS] = scaler.fit_transform(X_train[SOCIO_COLS])
    X_test_scaled[SOCIO_COLS] = scaler.transform(X_test[SOCIO_COLS])

    # --- Cross-validation (5 folds) avec GroupKFold pour éviter les fuites ---
    print(f"\n  Validation croisee ({args.cv_nsplits} folds) en cours...")
    gkf = GroupKFold(n_splits=args.cv_nsplits)
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train,
        groups=groups_train, 
        cv=gkf, scoring='accuracy'
    )
    cv_acc_mean = cv_scores.mean()
    cv_acc_std  = cv_scores.std()
    print(f"  CV Accuracy : {cv_acc_mean:.3f} +/- {cv_acc_std:.3f}")

    # --- Train and evaluate ---
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    class_names = [f"< {args.threshold}%", f">= {args.threshold}%"]
    print(f"\n  RESULTATS SUR LE TEST SET")
    print(f"  Accuracy : {test_acc:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    logs = {
        "task": "classification",
        "ml_method": args.ml_method,
        "indicator": "Symptoms of Depressive Disorder",
        "threshold": args.threshold,
        "n_train_states": n_train_states,
        "n_test_states": n_test_states,
        "cv_accuracy_mean": round(cv_acc_mean, 4),
        "cv_accuracy_std":  round(cv_acc_std, 4),
        "test_accuracy":    round(test_acc, 4)
    }

# =============================================================================
# TASK 3 : UNSUPERVISED ANALYSIS
# PCA + K-Means on state socioeconomic profiles
# =============================================================================
elif args.task == "clustering":
    print("\n" + "="*70)
    print("  TACHE 3 : ANALYSE NON SUPERVISEE (PCA + K-Means)")
    print("="*70)

    # Average profile per state across all years and indicators
    state_profile = df_merged.groupby('State').agg(
        mean_value=('Value', 'mean'),
        median_household_income=('median_household_income', 'mean'),
        poverty_rate=('poverty_rate', 'mean'),
        pct_bachelors_or_higher=('pct_bachelors_or_higher', 'mean'),
        pct_uninsured=('pct_uninsured', 'mean'),
        unemployment_rate=('unemployment_rate', 'mean')
    ).reset_index()

    unsup_features = ['mean_value'] + SOCIO_COLS
    X = state_profile[unsup_features].dropna()
    print(f"\n  Profils d'Etats : {X.shape[0]}  |  Features : {X.shape[1]}")
    print(f"  Features : {unsup_features}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- PCA ---
    print("\n  PCA -- Variance expliquee :")
    pca = PCA()
    pca.fit(X_scaled)
    var_exp = pca.explained_variance_ratio_
    var_cum = np.cumsum(var_exp)
    for i, (v, c) in enumerate(zip(var_exp, var_cum)):
        print(f"    PC{i+1} : {v:.3f}  (cumulee : {c:.3f})")

    # --- K-Means ---
    k = args.kmeans_k
    print(f"\n  K-Means (k={k}) :")
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    state_profile_valid = state_profile.loc[X.index].copy()
    state_profile_valid['cluster'] = labels

    sil = silhouette_score(X_scaled, labels)
    print(f"  Silhouette score : {sil:.3f}")

    print(f"\n  Etats par cluster :")
    cluster_summary = {}
    for c in range(k):
        states = state_profile_valid[state_profile_valid['cluster'] == c]['State'].tolist()
        mean_val = state_profile_valid[state_profile_valid['cluster'] == c]['mean_value'].mean()
        print(f"    Cluster {c} ({len(states)} Etats, taux moyen={mean_val:.1f}%) :")
        print(f"      {', '.join(states)}")
        cluster_summary[f"cluster_{c}"] = {
            "n_states": len(states),
            "mean_symptom_rate": round(mean_val, 2),
            "states": states
        }

    pipeline = None
    logs = {
        "task": "clustering",
        "kmeans_k": k,
        "silhouette_score": round(sil, 4),
        "pca_variance_explained": {
            f"PC{i+1}": round(v, 4) for i, v in enumerate(var_exp)
        },
        "clusters": cluster_summary
    }

# =============================================================================
# Save model and logs
# =============================================================================
print(f"\n>>> Sauvegarde des resultats dans : {out_dir}")

if args.task == "clustering":
    with open(path_model, 'wb') as f:
        pickle.dump({"scaler": scaler, "pca": pca, "kmeans": km}, f)
else:
    with open(path_model, 'wb') as f:
        pickle.dump({"scaler": scaler, "model": model}, f)

with open(path_logs, "w") as f:
    json.dump(logs, f, indent=2)

print(f"    config.json sauvegarde.")
print(f"    model.pkl   sauvegarde.")
print(f"    logs.json   sauvegarde.")
print(f"\n{'='*70}")
print(f"  Execution terminee.")
print(f"{'='*70}\n")
