"""
ML Model Training Script for Thyroid Risk Prediction
Trains, evaluates, and saves the predictive model with comprehensive metrics.
"""

import os
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# XGBoost is optional but preferred
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve

# import matplotlib.pyplot as plt
# import seaborn as sns

warnings.filterwarnings('ignore')


class ThyroidModelTrainer:
    def __init__(self, data_path: str = 'data/raw/Thyroid_Data.csv', output_dir: str = 'output'):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = None
        self.preprocessor = None
        self.model = None

        self.numeric_features = ['age', 'tsh', 't3', 'tt4', 't4u', 'fti']
        self.categorical_features = []

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.y_pred = None
        self.y_pred_proba = None

        self.metrics = {}

    def load_data(self):
        print('Loading data from', self.data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        self.df = pd.read_csv(self.data_path, na_values='?')
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
        print('Loaded dataset with shape', self.df.shape)
        return self

    def engineer_target(self):
        print('Engineering binary target using TSH thresholds')
        def derive(tsh):
            if pd.isna(tsh):
                return 0
            return 1 if (tsh > 4.5 or tsh < 0.45) else 0

        if 'target' not in self.df.columns:
            if 'tsh' in self.df.columns:
                self.df['target'] = self.df['tsh'].apply(derive)
            else:
                self.df['target'] = np.random.randint(0, 2, size=len(self.df))
        print('Target distribution:\n', self.df['target'].value_counts())
        return self

    def prepare_features(self):
        print('Preparing feature lists')
        # determine categorical features as remaining columns except index/target
        exclude = set(self.numeric_features + ['target', 'referral_source', 'patient_age'])
        self.categorical_features = [c for c in self.df.columns if c not in exclude]

        # keep only existing numeric features
        self.numeric_features = [c for c in self.numeric_features if c in self.df.columns]
        print('Numeric features:', self.numeric_features)
        print('Categorical features:', self.categorical_features)
        return self

    def build_preprocessor(self):
        print('Building preprocessor')
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, self.numeric_features),
            ('cat', categorical_transformer, self.categorical_features)
        ], remainder='drop')

        return self

    def split_data(self, test_size=0.2, random_state=42):
        print('Splitting data')
        features = self.numeric_features + self.categorical_features
        X = self.df[features]
        y = self.df['target']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        print('Train/Test sizes:', self.X_train.shape, self.X_test.shape)
        return self

    def fit_preprocessor(self):
        print('Fitting preprocessor on training data')
        self.preprocessor.fit(self.X_train)
        return self

    def train_models(self):
        print('Training models')
        X_train_t = self.preprocessor.transform(self.X_train)
        X_test_t = self.preprocessor.transform(self.X_test)

        # Ensure dense arrays for sklearn estimators that expect dense input
        def _to_dense(x):
            if hasattr(x, 'toarray'):
                return x.toarray()
            return x

        X_train_t = _to_dense(X_train_t)
        X_test_t = _to_dense(X_test_t)

        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        }
        if _HAS_XGB:
            models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)

        best_auc = -1
        best_name = None

        for name, m in models.items():
            print('Training', name)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            try:
                scores = cross_val_score(m, X_train_t, self.y_train, cv=cv, scoring='roc_auc')
                print(f'  CV AUC: {scores.mean():.4f} (+/- {scores.std():.4f})')
            except Exception as e:
                print('  CV failed:', e)

            m.fit(X_train_t, self.y_train)
            proba = m.predict_proba(X_test_t)[:, 1]
            pred = m.predict(X_test_t)
            auc_score = roc_auc_score(self.y_test, proba)
            f1 = f1_score(self.y_test, pred)
            print(f'  Test AUC: {auc_score:.4f}, F1: {f1:.4f}')

            if auc_score > best_auc:
                best_auc = auc_score
                best_name = name
                self.model = m
                self.y_pred = pred
                self.y_pred_proba = proba

        print('Best model:', best_name, 'AUC:', best_auc)
        self.metrics['best_model'] = best_name
        self.metrics['best_auc'] = float(best_auc)
        return self

    def evaluate_model(self):
        print('Evaluating model')
        # Classification report
        cr = classification_report(self.y_test, self.y_pred, output_dict=True)
        self.metrics['classification_report'] = cr
        # Confusion
        cm = confusion_matrix(self.y_test, self.y_pred).tolist()
        self.metrics['confusion_matrix'] = cm
        # AUC / PR
        auc_score = roc_auc_score(self.y_test, self.y_pred_proba)
        pr = average_precision_score(self.y_test, self.y_pred_proba)
        self.metrics['roc_auc'] = float(auc_score)
        self.metrics['average_precision'] = float(pr)

        print('ROC AUC:', auc_score)
        print('Average precision:', pr)
        return self

    def save_artifacts(self):
        print('Saving artifacts to', self.output_dir)
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)

        with open(models_dir / 'preprocessor.pkl', 'wb') as f:
            pickle.dump(self.preprocessor, f)
        with open(models_dir / 'model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open(models_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print('Artifacts saved')
        return self

    def run_full_pipeline(self):
        return (
            self.load_data()
                .engineer_target()
                .prepare_features()
                .build_preprocessor()
                .split_data()
                .fit_preprocessor()
                .train_models()
                .evaluate_model()
                .save_artifacts()
        )


if __name__ == '__main__':
    trainer = ThyroidModelTrainer()
    trainer.run_full_pipeline()
