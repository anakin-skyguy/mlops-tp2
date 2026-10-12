# TP2 MLOps : du modèle ML à la production

Mini-projet EMSI (Prof. Bakkali Yedri Othman) : **Données → Training → MLflow → Modèle → API → Docker → Monitoring → Drift**.

Le modèle est un `RandomForestClassifier` entraîné sur le dataset Iris (fourni par scikit-learn), exposé par une API Flask, conteneurisé avec Docker, testé automatiquement par une CI GitHub Actions, et surveillé par un script de détection de data drift.

## Structure du dépôt

```
mlops-tp2/
├── .github/workflows/ci.yml   # CI : installe les dépendances et lance pytest
├── models/iris_model.pkl      # modèle entraîné (joblib)
├── screenshots/               # preuves (MLflow, API, Docker, drift, tests, CI)
├── src/
│   ├── train.py               # entraînement + suivi MLflow
│   ├── predict.py             # prédiction en ligne de commande
│   ├── api.py                 # API Flask (/, /health, /predict)
│   └── drift.py               # détection simple de data drift
├── tests/test_api.py          # 4 tests pytest de l'API
├── Dockerfile
├── requirements.txt           # versions figées (identiques à l'entraînement)
└── .dockerignore
```

## Prérequis (Windows)

Python 3.12, Git et Docker Desktop.

```
python --version
git --version
docker --version
```

## Installation

```
git clone https://github.com/anakin-skyguy/mlops-tp2.git
cd mlops-tp2
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scikit-learn mlflow joblib flask pytest
```

Toutes les commandes se lancent **depuis la racine du projet** (les chemins sont relatifs).

## 1. Entraînement et suivi MLflow

```
python src/train.py 50
python src/train.py 100
python src/train.py 200
mlflow ui --port 5001
```

L'argument est le nombre d'arbres (`n_estimators`, 100 par défaut). L'interface est sur http://127.0.0.1:5001, onglet **Model training**, expérience `Iris_MLOps`. Le port 5001 évite le conflit avec l'API (5000).

Note : avec MLflow 3.17, l'enregistrement du modèle scikit-learn exige de déclarer le type `sklearn.tree._tree.Tree` comme fiable (`skops_trusted_types`). C'est fait dans `train.py`.

### Résultats

| Run | n_estimators | accuracy |
|---|---|---|
| charming-pig-251 | 50 | 0.9 |
| wistful-elk-585 | 100 | 0.9 |
| unleashed-mink-639 | 200 | 0.9 |

Conclusion : augmenter le nombre d'arbres n'améliore pas la précision sur ce jeu de données. 50 arbres suffisent et coûtent moins cher. Attention : le jeu de test ne contient que 30 fleurs (une fleur = 3,3 points de précision), l'écart entre deux modèles resterait donc difficile à interpréter même s'il apparaissait. Une validation croisée serait plus fiable.

Captures : `screenshots/01-mlflow-chart.png` et `screenshots/02-mlflow-table.png`.

## 2. Prédiction

```
python src/predict.py
```

Résultat attendu : `Prediction: 0 setosa`.

## 3. API

```
python src/api.py
```

| Route | Méthode | Rôle |
|---|---|---|
| `/` | GET | état du service |
| `/health` | GET | `{"status": "ok", "model_version": "1.0"}` |
| `/predict` | POST | `{"features": [5.1, 3.5, 1.4, 0.2]}` renvoie la classe prédite et la version du modèle |

Une entrée invalide (liste qui n'a pas 4 valeurs) renvoie une erreur HTTP 400.

Exemple (PowerShell) :

```
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body '{"features":[5.1,3.5,1.4,0.2]}'
```

Capture : `screenshots/03-api-predict.webp`.

## 4. Docker

```
docker build -t mlops-tp2:1.0 .
docker run --name mlops-tp2 -p 5000:5000 mlops-tp2:1.0
```

L'image n'installe que pandas, numpy, scikit-learn, joblib et flask, en versions figées : un modèle sérialisé avec une version de scikit-learn peut ne pas se charger avec une autre. MLflow est volontairement absent de l'image.

Capture : `screenshots/04-docker-health-ps.png` (`/health` répond depuis le conteneur, `docker ps` montre `mlops-tp2` actif).

## 5. Tests et CI

```
python -m pytest -v
```

4 tests : `/health`, prédiction setosa (0), prédiction virginica (2), entrée invalide (400). Résultat : **4 passed**.

La CI (`.github/workflows/ci.yml`) lance ces mêmes tests à chaque push sur une machine Linux propre. Run #1 (commit `d4b522e`) : succès en 37 s.

Captures : `screenshots/07-pytest.png` et `screenshots/08-github-ci-green.png`.

## 6. Monitoring et data drift

```
python src/drift.py        # seuil par défaut : 1.0 écart-type
python src/drift.py 2.0    # seuil plus strict
```

Le script compare les données d'entraînement (Iris) à de **nouvelles données simulées** : longueur des pétales × 1,5 et largeur des sépales × 0,8. Ces données sont **synthétiques**, ce n'est pas de la donnée réelle de production. Pour chaque variable, il calcule le décalage de la moyenne en écarts-types d'entraînement et le pourcentage de valeurs hors de la plage vue à l'entraînement.

| Variable | Moyenne (train → new) | Décalage (écarts-types) | Valeurs hors plage |
|---|---|---|---|
| sepal length | 5.84 → 5.84 | 0.00 | 0 % |
| sepal width | 3.06 → 2.45 | 1.40 | 7.33 % |
| petal length | 3.76 → 5.64 | 1.06 | 40.00 % |
| petal width | 1.20 → 1.20 | 0.00 | 0 % |

Avec le seuil 1.0, le script affiche une **ALERTE** sur sepal width et petal length. Avec le seuil 2.0, il affiche **OK**, alors que les données sont identiques.

Captures : `screenshots/05-drift-alert-1.0.png` et `screenshots/06-drift-ok-2.0.png`.

### Synthèse Data Drift

**Quelle variable change ?** La longueur des pétales (moyenne 3.76 → 5.64 cm, maximum 6.9 → 10.35 cm, écart-type 1.77 → 2.65) et, dans une moindre mesure, la largeur des sépales (3.06 → 2.45 cm). La longueur des sépales et la largeur des pétales sont inchangées.

**Quel impact ?** Le modèle apprend des seuils sur les mesures des pétales. Avec 40 % des longueurs de pétales hors de la plage d'entraînement, il extrapole : ses prédictions deviennent peu fiables, et il peut pousser des fleurs vers la classe virginica. Le danger est que l'API continue de renvoyer une classe sans erreur, donc la dégradation de précision est silencieuse.

**Quand réentraîner ?** Quand l'alerte de drift se déclenche **et** que la précision mesurée sur des données récentes étiquetées passe sous un plancher convenu à l'avance (par exemple 5 points sous les 0.90 de référence), ou de façon planifiée. On réentraîne alors avec des données incluant la nouvelle distribution, on enregistre un nouveau run MLflow, on compare avec l'ancien, puis on incrémente `MODEL_VERSION` et on redéploie.

**Limite du seuil.** À 2.0 écarts-types, une règle basée sur la moyenne a manqué un drift pourtant massif (+50 % sur la longueur des pétales). Le seuil est un compromis entre fausses alertes et dérives manquées. Le pourcentage de valeurs hors plage, qui atteint 40 %, détecte ce que la moyenne seule ne voit pas, ce qui justifie de combiner plusieurs indicateurs (moyenne, écart-type, plage, tests statistiques).

## 7. Comparaison DevOps / MLOps

| | DevOps | MLOps |
|---|---|---|
| Ce qui évolue | Code | Code + **données** + **modèle** |
| Pipeline | Tests, Git, CI/CD, Docker, monitoring | Idem + training, suivi d'expériences (MLflow), CT (entraînement continu) |
| Monitoring | Disponibilité, erreurs, latence | Idem + **qualité du modèle**, data drift |
| Réaction aux problèmes | Corriger le code, redéployer | Idem + **réentraîner** avec de nouvelles données |
| Dans ce TP | Git, Docker, pytest, GitHub Actions | MLflow, `models/`, `drift.py`, `MODEL_VERSION` |

La différence clé : un service ML peut se dégrader sans qu'une ligne de code ne change, simplement parce que les données ont changé.

## Pistes d'amélioration

- Validation croisée au lieu d'un seul jeu de test de 30 fleurs.
- Réentraînement automatique après une alerte de drift, avec comparaison MLflow et passage à `MODEL_VERSION = "2.0"`.
- Test de drift statistique (Kolmogorov-Smirnov, PSI) en plus du décalage de moyenne.
- Intégrer `drift.py` à la CI ou à une tâche planifiée.