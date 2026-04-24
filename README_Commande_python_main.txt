## Contexte
Bienvenue dans ce projet ! Le code d'exploration et l'analyse détaillée se trouvent principalement dans le **Notebook**. 

Le point d'entrée principal du projet est le script `main.py`. Il est conçu pour être exécuté en ligne de commande et permet de lancer facilement les différents modèles disponibles.

##Utilisation (Commandes Python)

Pour exécuter les modèles, utilisez les commandes suivantes depuis la racine du projet :

# Regression (7 modeles)
python main.py --task regression --ml_method knn --knn_n_neighbors 30  # Meilleur score
python main.py --task regression --ml_method svr
python main.py --task regression --ml_method ridge
python main.py --task regression --ml_method linear_reg
python main.py --task regression --ml_method sgd
python main.py --task regression --ml_method random_forest
python main.py --task regression --ml_method gradient_boosting

# Classification (6 modeles)
python main.py --task classification --ml_method random_forest  # Meilleur score
python main.py --task classification --ml_method knn --knn_n_neighbors 20
python main.py --task classification --ml_method svm
python main.py --task classification --ml_method log_reg
python main.py --task classification --ml_method gnb
python main.py --task classification --ml_method gradient_boosting

# Clustering
python main.py --task clustering --kmeans_k 3
