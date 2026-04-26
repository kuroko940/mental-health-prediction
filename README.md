# Analyse de donnees sur la Sante Mentale

## Contexte
Bienvenue dans ce projet d'analyse de donnees sur la sante mentale. L'exploration initiale, le traitement des donnees et l'analyse detaillee se trouvent dans le notebook `exploration_modeles.ipynb`. 

Le point d'entree principal du projet est le script `main.py`. Il est concu pour etre execute en ligne de commande et permet de lancer facilement les differents modeles d'apprentissage automatique disponibles. Les resultats detailles et les conclusions sont documentes dans `rapport_technique.pdf`.

## 💾 Donnees (Datasets)
Ce projet croise des donnees de sante mentale avec des indicateurs socio-economiques. 

Les datasets socio-economiques (`master_socioeconomic_data_2020_2024.csv` et `state_socioeconomic_data.csv`) ont ete crees specifiquement pour ce projet et sont inclus directement dans le dossier `data/` de ce depot.

Cependant, le dataset principal sur la sante mentale provenant de Kaggle n'est pas inclus pour des raisons de taille.

**Pour executer le code localement :**
1. Telechargez le dataset `Indicators_of_Anxiety_or_Depression.csv` depuis Kaggle : [https://www.kaggle.com/datasets/melissamonfared/indicators-of-anxiety-or-depression]
2. Placez ce fichier telecharge dans le dossier `data/` de ce projet (a cote des deux autres fichiers deja presents).

Le script `main.py` et le notebook liront alors l'ensemble des donnees correctement.

## Utilisation

Pour executer les modeles, copiez et collez les commandes suivantes depuis la racine du projet dans votre terminal :

```bash
# --- REGRESSION (7 modeles) : Predit une valeur continue ---
# Meilleur score : KNN
python main.py --task regression --ml_method knn --knn_n_neighbors 30
python main.py --task regression --ml_method svr
python main.py --task regression --ml_method ridge
python main.py --task regression --ml_method linear_reg
python main.py --task regression --ml_method sgd
python main.py --task regression --ml_method random_forest
python main.py --task regression --ml_method gradient_boosting

# --- CLASSIFICATION (6 modeles) : Predit une categorie ---
# Meilleur score : Random Forest
python main.py --task classification --ml_method random_forest
python main.py --task classification --ml_method knn --knn_n_neighbors 20
python main.py --task classification --ml_method svm
python main.py --task classification --ml_method log_reg
python main.py --task classification --ml_method gnb
python main.py --task classification --ml_method gradient_boosting

# --- CLUSTERING : Regroupe les profils similaires ---
python main.py --task clustering --kmeans_k 3
