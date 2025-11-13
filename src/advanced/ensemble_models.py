"""
Ensemble ML Models: XGBoost + LightGBM + CatBoost

Combines multiple gradient boosting algorithms for robust predictions.

References:
- Chen & Guestrin (2016) - XGBoost
- Ke et al. (2017) - LightGBM
- Prokhorenkova et al. (2018) - CatBoost
- Zhou (2012) - Ensemble Methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble of XGBoost + LightGBM + CatBoost"""

    def __init__(self, config: Dict, ensemble_method: str = 'weighted'):
        """
        Initialize ensemble predictor

        Args:
            config: Configuration dictionary
            ensemble_method: 'weighted', 'voting', or 'stacking'
        """
        self.config = config
        self.ensemble_method = ensemble_method

        # Individual models
        self.xgb_model = None
        self.lgb_model = None
        self.cat_model = None

        # Model weights (learned from validation performance)
        self.weights = {'xgb': 0.33, 'lgb': 0.33, 'cat': 0.34}

        # Scaler
        self.scaler = StandardScaler()
        self.feature_names = []

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train all ensemble models

        Args:
            features: Feature DataFrame
            target: Target Series
            validation_split: Validation set size
            verbose: Print progress

        Returns:
            Dictionary with metrics for each model
        """
        logger.info("Training ensemble models (XGB + LGB + CAT)...")

        self.feature_names = list(features.columns)

        # Prepare data
        X, y = self._prepare_data(features, target)

        # Split data (preserve time order)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

        # Train XGBoost
        logger.info("Training XGBoost...")
        xgb_metrics = self._train_xgboost(X_train, y_train, X_val, y_val, verbose)

        # Train LightGBM
        logger.info("Training LightGBM...")
        lgb_metrics = self._train_lightgbm(X_train, y_train, X_val, y_val, verbose)

        # Train CatBoost
        logger.info("Training CatBoost...")
        cat_metrics = self._train_catboost(X_train, y_train, X_val, y_val, verbose)

        # Calculate optimal weights based on validation performance
        self._calculate_optimal_weights(xgb_metrics, lgb_metrics, cat_metrics)

        # Ensemble metrics
        ensemble_metrics = self._evaluate_ensemble(X_val, y_val)

        all_metrics = {
            'xgboost': xgb_metrics,
            'lightgbm': lgb_metrics,
            'catboost': cat_metrics,
            'ensemble': ensemble_metrics
        }

        if verbose:
            self._print_comparison(all_metrics)

        return all_metrics

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        verbose: bool
    ) -> Dict[str, float]:
        """Train XGBoost model"""
        # Handle class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        self.xgb_model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            objective='binary:logistic',
            eval_metric='auc',
            early_stopping_rounds=10,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )

        eval_set = [(X_train, y_train), (X_val, y_val)]
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        # Evaluate
        y_pred = self.xgb_model.predict(X_val)
        y_pred_proba = self.xgb_model.predict_proba(X_val)[:, 1]

        return {
            'accuracy': accuracy_score(y_val, y_pred),
            'auc': roc_auc_score(y_val, y_pred_proba)
        }

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        verbose: bool
    ) -> Dict[str, float]:
        """Train LightGBM model"""
        # Calculate class weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        self.lgb_model = lgb.LGBMClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            objective='binary',
            metric='auc',
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1
        )

        self.lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )

        # Evaluate
        y_pred = self.lgb_model.predict(X_val)
        y_pred_proba = self.lgb_model.predict_proba(X_val)[:, 1]

        return {
            'accuracy': accuracy_score(y_val, y_pred),
            'auc': roc_auc_score(y_val, y_pred_proba)
        }

    def _train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        verbose: bool
    ) -> Dict[str, float]:
        """Train CatBoost model"""
        # Calculate class weight
        class_counts = y_train.value_counts()
        class_weights = {0: 1.0, 1: class_counts[0] / class_counts[1]}

        self.cat_model = cb.CatBoostClassifier(
            depth=6,
            learning_rate=0.1,
            iterations=100,
            objective='Logloss',
            eval_metric='AUC',
            early_stopping_rounds=10,
            class_weights=class_weights,
            random_state=42,
            verbose=False
        )

        self.cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )

        # Evaluate
        y_pred = self.cat_model.predict(X_val)
        y_pred_proba = self.cat_model.predict_proba(X_val)[:, 1]

        return {
            'accuracy': accuracy_score(y_val, y_pred),
            'auc': roc_auc_score(y_val, y_pred_proba)
        }

    def _calculate_optimal_weights(
        self,
        xgb_metrics: Dict,
        lgb_metrics: Dict,
        cat_metrics: Dict
    ):
        """Calculate optimal ensemble weights based on validation AUC"""
        total_auc = (
            xgb_metrics['auc'] +
            lgb_metrics['auc'] +
            cat_metrics['auc']
        )

        # Weight proportional to AUC (better models get more weight)
        self.weights = {
            'xgb': xgb_metrics['auc'] / total_auc,
            'lgb': lgb_metrics['auc'] / total_auc,
            'cat': cat_metrics['auc'] / total_auc
        }

        logger.info(f"Optimal weights: XGB={self.weights['xgb']:.3f}, "
                   f"LGB={self.weights['lgb']:.3f}, "
                   f"CAT={self.weights['cat']:.3f}")

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using ensemble

        Args:
            features: Feature DataFrame

        Returns:
            Array of probabilities for class 1
        """
        X, _ = self._prepare_data(features, pd.Series(index=features.index))

        # Get predictions from each model
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
        cat_proba = self.cat_model.predict_proba(X)[:, 1]

        if self.ensemble_method == 'weighted':
            # Weighted average
            ensemble_proba = (
                xgb_proba * self.weights['xgb'] +
                lgb_proba * self.weights['lgb'] +
                cat_proba * self.weights['cat']
            )
        elif self.ensemble_method == 'voting':
            # Simple average
            ensemble_proba = (xgb_proba + lgb_proba + cat_proba) / 3
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

        return ensemble_proba

    def predict(self, features: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels"""
        proba = self.predict_proba(features)
        return (proba >= threshold).astype(int)

    def _evaluate_ensemble(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        y_pred_proba = self.predict_proba(X_val)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        return {
            'accuracy': accuracy_score(y_val, y_pred),
            'auc': roc_auc_score(y_val, y_pred_proba)
        }

    def _prepare_data(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare and scale data"""
        # Handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill().fillna(0)

        # Scale features
        if not hasattr(self.scaler, 'mean_'):
            # Fit scaler
            X = pd.DataFrame(
                self.scaler.fit_transform(features),
                columns=features.columns,
                index=features.index
            )
        else:
            # Transform only
            X = pd.DataFrame(
                self.scaler.transform(features),
                columns=features.columns,
                index=features.index
            )

        return X, target

    def _print_comparison(self, all_metrics: Dict):
        """Print model comparison"""
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)

        for model_name, metrics in all_metrics.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  AUC: {metrics['auc']:.4f}")

        logger.info("\n" + "="*60)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get aggregated feature importance from all models

        Args:
            top_n: Number of top features

        Returns:
            DataFrame with feature importance
        """
        # XGBoost importance
        xgb_imp = pd.DataFrame({
            'feature': self.feature_names,
            'xgb_importance': self.xgb_model.feature_importances_
        })

        # LightGBM importance
        lgb_imp = pd.DataFrame({
            'feature': self.feature_names,
            'lgb_importance': self.lgb_model.feature_importances_
        })

        # CatBoost importance
        cat_imp = pd.DataFrame({
            'feature': self.feature_names,
            'cat_importance': self.cat_model.feature_importances_
        })

        # Merge
        importance = xgb_imp.merge(lgb_imp, on='feature').merge(cat_imp, on='feature')

        # Calculate weighted average
        importance['avg_importance'] = (
            importance['xgb_importance'] * self.weights['xgb'] +
            importance['lgb_importance'] * self.weights['lgb'] +
            importance['cat_importance'] * self.weights['cat']
        )

        importance = importance.sort_values('avg_importance', ascending=False)

        return importance.head(top_n)
