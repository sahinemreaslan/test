"""
XGBoost Model for Trading Signal Prediction

Uses gradient boosting to predict trading signals based on:
- Multi-timeframe fractal patterns
- Technical indicators
- Market regime features
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib
import logging

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """XGBoost-based trading signal predictor"""

    def __init__(self, config: Dict):
        """
        Initialize XGBoost predictor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.xgb_config = config.get('xgboost', {})

        # Model parameters
        self.model_params = self.xgb_config.get('model_params', {})
        self.train_params = self.xgb_config.get('train_params', {})

        # Model and scaler
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None

        # Feature names
        self.feature_names: List[str] = []

        # Training history
        self.training_history = {}

    def prepare_data(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        scale: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training

        Args:
            features: Feature DataFrame
            target: Target Series
            scale: Whether to scale features

        Returns:
            Tuple of (scaled features, target)
        """
        # Store feature names
        self.feature_names = list(features.columns)

        # Handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Scale features
        if scale:
            if self.scaler is None:
                self.scaler = StandardScaler()
                features_scaled = pd.DataFrame(
                    self.scaler.fit_transform(features),
                    columns=features.columns,
                    index=features.index
                )
            else:
                features_scaled = pd.DataFrame(
                    self.scaler.transform(features),
                    columns=features.columns,
                    index=features.index
                )
        else:
            features_scaled = features

        return features_scaled, target

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        validation_split: float = 0.2,
        use_time_series_split: bool = True,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train XGBoost model

        Args:
            features: Feature DataFrame
            target: Target Series
            validation_split: Validation set size
            use_time_series_split: Use time series cross-validation
            verbose: Print training progress

        Returns:
            Dictionary with training metrics
        """
        logger.info("Training XGBoost model...")

        # Prepare data
        X, y = self.prepare_data(features, target, scale=True)

        if use_time_series_split:
            # Time series cross-validation
            metrics = self._train_with_time_series_cv(X, y, verbose=verbose)
        else:
            # Simple train/test split
            metrics = self._train_with_simple_split(X, y, validation_split, verbose=verbose)

        logger.info("Training complete")

        return metrics

    def _train_with_simple_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float,
        verbose: bool
    ) -> Dict[str, float]:
        """Train with simple train/test split"""
        # Split data (preserve time order)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
        logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Val target distribution: {y_val.value_counts().to_dict()}")

        # Handle class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")

        # Initialize model
        self.model = xgb.XGBClassifier(
            **self.model_params,
            scale_pos_weight=scale_pos_weight
        )

        # Train
        eval_set = [(X_train, y_train), (X_val, y_val)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )

        # Evaluate
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]

        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)

        if verbose:
            logger.info(f"Validation Metrics: {metrics}")

        return metrics

    def _train_with_time_series_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool
    ) -> Dict[str, float]:
        """Train with time series cross-validation"""
        n_splits = self.train_params.get('cv_folds', 5)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv_metrics = []

        logger.info(f"Running {n_splits}-fold time series cross-validation...")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"Fold {fold}/{n_splits}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Handle class imbalance
            if (y_train == 1).sum() > 0:
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            else:
                scale_pos_weight = 1.0

            # Train model for this fold
            model = xgb.XGBClassifier(
                **self.model_params,
                scale_pos_weight=scale_pos_weight
            )

            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

            # Evaluate
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            fold_metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
            cv_metrics.append(fold_metrics)

            if verbose:
                logger.info(f"  Fold {fold} - Accuracy: {fold_metrics['accuracy']:.4f}, "
                          f"AUC: {fold_metrics['roc_auc']:.4f}")

        # Average metrics across folds
        avg_metrics = {
            metric: np.mean([m[metric] for m in cv_metrics])
            for metric in cv_metrics[0].keys()
        }

        logger.info(f"Average CV Metrics: {avg_metrics}")

        # Train final model on all data (with validation set for early stopping)
        scale_pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0

        # Split for validation (last 20%)
        split_idx = int(len(X) * 0.8)
        X_train_final = X.iloc[:split_idx]
        y_train_final = y.iloc[:split_idx]
        X_val_final = X.iloc[split_idx:]
        y_val_final = y.iloc[split_idx:]

        self.model = xgb.XGBClassifier(
            **self.model_params,
            scale_pos_weight=scale_pos_weight
        )

        eval_set = [(X_train_final, y_train_final), (X_val_final, y_val_final)]
        self.model.fit(X_train_final, y_train_final, eval_set=eval_set, verbose=verbose)

        return avg_metrics

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
        }

        return metrics

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict trading signals

        Args:
            features: Feature DataFrame

        Returns:
            Array of predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Prepare features
        X, _ = self.prepare_data(features, pd.Series(index=features.index), scale=True)

        # Predict
        predictions = self.model.predict(X)

        return predictions

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict signal probabilities

        Args:
            features: Feature DataFrame

        Returns:
            Array of probabilities for class 1
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Prepare features
        X, _ = self.prepare_data(features, pd.Series(index=features.index), scale=True)

        # Predict probabilities
        probabilities = self.model.predict_proba(X)[:, 1]

        return probabilities

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance.head(top_n)

    def save_model(self, filepath: str):
        """
        Save model to disk

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model from disk

        Args:
            filepath: Path to load model from
        """
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']

        logger.info(f"Model loaded from {filepath}")

    def get_model_summary(self) -> Dict:
        """Get model summary information"""
        if self.model is None:
            return {'status': 'not trained'}

        return {
            'status': 'trained',
            'n_features': len(self.feature_names),
            'model_params': self.model.get_params(),
            'feature_importance_top_5': self.get_feature_importance(top_n=5).to_dict('records')
        }
