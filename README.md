# ONCFM — Détection de faux billets par Machine Learning

Mission senior Data Analyst pour l'Organisation nationale de lutte contre le faux-monnayage (ONCFM), visant à fournir une application de machine learning capable de prédire si un billet en euros est vrai ou faux.

---

## Contexte / besoin métier

L'ONCFM met en place des méthodes d'identification des faux billets en euros pour lutter contre la contrefaçon. Marie, responsable de la lutte contre le faux-monnayage, souhaite mettre à disposition des équipes une application de machine learning : après le scan d'un billet (longueur, hauteur, largeur, etc.), l'application doit prédire s'il s'agit d'un vrai ou d'un faux billet.

L'ONCFM ne disposant pas des compétences en interne pour développer cette solution, la mission est confiée en autonomie complète sur la partie technique. L'agence européenne EMV (European Monetary Verification) recommande de tester en priorité quatre algorithmes : K-means, régression logistique, KNN et Random Forest.

## Données (source, qualité, limites)

**Sources :** jeu de données de 1 500 billets scannés, avec 6 caractéristiques géométriques par billet (`diagonal`, `height_left`, `height_right`, `margin_low`, `margin_up`, `length`) et le label `is_genuine` (vrai/faux).

**Qualité :**
- Après suppression des lignes à valeurs manquantes, l'échantillon exploitable passe de 1 500 à **1 463 billets** (971 vrais, 492 faux).
- La heatmap de corrélation entre les 6 variables montre qu'aucune n'est redondante : chacune apporte une information distincte, aucune ne domine ou n'écrase les autres.
- L'analyse univariée montre que `length` et `margin_low` sont les deux variables qui différencient le plus nettement les vrais des faux billets.

**Limites :**
- Les billets à données manquantes ont été purement exclus de l'analyse plutôt qu'imputés, ce qui réduit l'échantillon de test disponible sans chercher à réintégrer ces cas.
- Le fichier de sauvegarde du modèle produit par le notebook d'analyse (`random_forest_billets.pkl`) ne correspond pas au fichier chargé par le script applicatif final (`regression_logistique_billets.pkl`) — la régression logistique déployée doit avoir été réentraînée et sauvegardée séparément ; ce point mérite d'être vérifié/documenté pour assurer la traçabilité entre le notebook d'analyse et l'application livrée.

## Démarche (choix, outils, étapes)

1. Chargement et nettoyage des données (suppression des valeurs manquantes), exploration statistique (moyennes, écarts-types) et analyse de corrélation entre les 6 variables.
2. Analyse univariée par variable pour identifier celles qui distinguent le mieux les vrais des faux billets.
3. Séparation des données en jeu d'entraînement (80 %, 1 170 billets) et jeu de test (20 %, 293 billets), avec validation croisée à 5 itérations pour optimiser la répartition.
4. Standardisation des variables, puis entraînement et évaluation des **4 modèles recommandés par l'EMV** : régression logistique, K-means, KNN (5 voisins), Random Forest (100 arbres).
5. Comparaison des modèles sur l'accuracy globale et surtout sur le **rappel de la classe « faux billet »** (prioritaire pour des raisons économiques : un vrai billet mal classé coûte bien moins cher qu'un faux billet non détecté), via matrices de confusion et courbes ROC.
6. Sélection du modèle final selon un arbitrage performance / coût / rapidité / interprétabilité.
7. Sauvegarde du modèle et du scaler (`joblib`), puis développement d'un script applicatif séparé chargeant le modèle pour prédire la nature de nouveaux billets à partir d'un fichier CSV.

**Outil :** Python (pandas, scikit-learn, seaborn/matplotlib, joblib).

## Résultats + impact / recommandations

**Performances des 4 modèles sur le jeu de test (293 billets, dont 99 faux) :**

| Modèle | Accuracy | Faux billets détectés (rappel) |
|---|---|---|
| Régression logistique | 99,66 % | 100 % |
| Random Forest | 99,32 % | 100 % |
| K-means | 98,98 % | 98,99 % |
| KNN | 97,98 % | 97,98 % |

- Deux modèles se détachent avec **100 % de détection des faux billets** : la régression logistique et le Random Forest.
- **Modèle final retenu : la régression logistique** — à performance quasi équivalente au Random Forest sur la détection des faux, elle est la moins coûteuse en calcul, la plus rapide et la plus interprétable, des critères jugés déterminants pour un déploiement opérationnel chez l'ONCFM.
- **Application fonctionnelle livrée** : un script charge le modèle de régression logistique entraîné et le scaler associé, lit un fichier CSV de billets à tester, et retourne pour chacun la prédiction (vrai/faux) ainsi que la probabilité associée — par exemple, un billet testé est classé faux avec 99,68 % de confiance, un autre authentique avec 99,93 % de confiance.
- **Impact attendu :** donner aux équipes de l'ONCFM un outil opérationnel, rapide et fiable pour accélérer l'identification des faux billets, avec une garantie de détection de 100 % des faux sur le jeu de test.

## Limites + prochaines pistes

- Les performances mesurées (100 % de rappel sur les faux) le sont sur un jeu de test de seulement 99 faux billets ; une validation sur un échantillon plus large ou de nouveaux billets réels renforcerait la confiance dans ce résultat avant généralisation.
- L'incohérence entre le fichier de modèle sauvegardé dans le notebook d'analyse (Random Forest) et celui utilisé par le script applicatif (régression logistique) doit être clarifiée pour garantir que le modèle réellement déployé correspond bien à celui documenté et évalué dans la présentation.
- Le modèle est entraîné sur un échantillon figé de billets scannés ; sa robustesse face à de nouvelles techniques de contrefaçon devra être réévaluée périodiquement avec de nouvelles données.

---

*Projet réalisé dans le cadre de la mission senior Data Analyst pour l'ONCFM, sous la responsabilité de Marie, responsable de la lutte contre le faux-monnayage.*
