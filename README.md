# ORANGE SUMMER CHALLENGE - AI BUSNESS FOR ACCELERATOR

# Prédiction du Churn Client

Projet d'analyse et de prédiction de la résiliation client (churn) à partir de données télécom.

## Données

915 clients, 12 variables : client_id, anciennete_mois, age, region, type_offre, moyen_paiement, conso_data_go, conso_voix_min, conso_sms, nb_contacts_support, nb_incidents_reseau, churn (cible).

## Installation

pip install pandas numpy matplotlib seaborn scikit-learn joblib
## Bibliothèques utilisées

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
## 1. Nettoyage des données

- Suppression des doublons (drop_duplicates)
- Correction des incohérences de saisie (casse, espaces)
- Imputation des valeurs manquantes (médiane pour les numériques, mode pour les catégorielles)
- Traitement des valeurs aberrantes par winsorisation (méthode IQR)

Résultat : dataset_propre.csv

## 2. Analyse exploratoire

Le churn est rare (~12% des clients) et touche surtout les clients avec une faible ancienneté et beaucoup de contacts support.

## 3. Modélisation

- Modèle : Random Forest Classifier (class_weight='balanced')
- Préparation : encodage des variables catégorielles, split train/test (80/20) stratifié
- Variables les plus influentes : ancienneté, consommation data, consommation voix

## 4. Évaluation

Métriques : accuracy, precision, recall, F1-score, AUC-ROC.
