"""
Priority Model for AI Task Management System
Predicts task priority scores based on multiple factors
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class TaskPriorityModel:
    """Model for predicting task priority scores"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize priority model
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'linear')
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                min_samples_split=5,
                min_samples_leaf=2
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def calculate_priority_score(self, 
                                urgency_score: float,
                                complexity_score: float,
                                days_until_deadline: int,
                                business_impact: float = 5.0) -> float:
        """
        Calculate priority score using rule-based approach
        This can serve as training labels or fallback
        
        Args:
            urgency_score: Urgency level (0-10)
            complexity_score: Task complexity (0-10)
            days_until_deadline: Days until deadline
            business_impact: Business impact score (0-10)
        
        Returns:
            Priority score (0-10)
        """
        # Deadline urgency factor
        if days_until_deadline < 0:
            deadline_factor = 10  # Overdue
        elif days_until_deadline == 0:
            deadline_factor = 9   # Due today
        elif days_until_deadline <= 3:
            deadline_factor = 8   # Due very soon
        elif days_until_deadline <= 7:
            deadline_factor = 6   # Due this week
        elif days_until_deadline <= 30:
            deadline_factor = 4   # Due this month
        else:
            deadline_factor = 2   # Future deadline
        
        # Weighted priority calculation
        priority = (
            urgency_score * 0.3 +
            deadline_factor * 0.4 +
            business_impact * 0.2 +
            (10 - complexity_score) * 0.1  # Less complex = higher priority for quick wins
        )
        
        return max(0, min(10, priority))
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction"""
        feature_columns = [
            'urgency_score',
            'complexity_score',
            'days_until_deadline',
            'deadline_urgency_multiplier',
            'estimated_hours'
        ]
        
        # Add encoded categorical features if they exist
        categorical_encoded = [col for col in df.columns if col.endswith('_encoded')]
        feature_columns.extend(categorical_encoded)
        
        # Create additional engineered features
        df_features = df.copy()
        
        # Task urgency-complexity ratio
        df_features['urgency_complexity_ratio'] = (
            df_features['urgency_score'] / (df_features['complexity_score'] + 1)
        )
        
        # Deadline pressure score
        df_features['deadline_pressure'] = (
            df_features['deadline_urgency_multiplier'] * 
            (1 / (df_features['days_until_deadline'] + 1))
        )
        
        # Effort vs impact score
        df_features['effort_impact_score'] = (
            df_features['urgency_score'] / (df_features['estimated_hours'] + 1)
        )
        
        feature_columns.extend([
            'urgency_complexity_ratio',
            'deadline_pressure', 
            'effort_impact_score'
        ])
        
        # Select only available features
        available_features = [col for col in feature_columns if col in df_features.columns]
        
        self.feature_columns = available_features
        return df_features[available_features]
    
    def generate_training_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate priority labels for training using rule-based approach"""
        labels = []
        
        for _, row in df.iterrows():
            urgency = row.get('urgency_score', 5)
            complexity = row.get('complexity_score', 5)
            days_until = row.get('days_until_deadline', 30)
            
            # Estimate business impact based on category or keywords
            business_impact = 5.0  # Default
            if 'category' in row:
                category = str(row['category']).lower()
                if any(word in category for word in ['critical', 'security', 'bug']):
                    business_impact = 8.0
                elif any(word in category for word in ['feature', 'enhancement']):
                    business_impact = 6.0
                elif any(word in category for word in ['maintenance', 'documentation']):
                    business_impact = 3.0
            
            priority = self.calculate_priority_score(
                urgency, complexity, days_until, business_impact
            )
            labels.append(priority)
        
        return np.array(labels)
    
    def train(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, float]:
        """
        Train the priority model
        
        Args:
            df: Training dataframe
            target_column: Column name for target variable (if None, generates labels)
        
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.model_type} priority model")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Prepare target
        if target_column and target_column in df.columns:
            y = df[target_column].values
        else:
            logger.info("Generating priority labels using rule-based approach")
            y = self.generate_training_labels(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        logger.info(f"Model training completed. R² score: {metrics['r2']:.3f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict priority scores for new tasks"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = self.prepare_features(df)
        
        # Ensure features match training features
        if self.feature_columns:
            missing_features = set(self.feature_columns) - set(X.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Add missing features with default values
                for feature in missing_features:
                    X[feature] = 0
            
            # Reorder columns to match training
            X = X[self.feature_columns]
        
        predictions = self.model.predict(X)
        
        # Ensure predictions are in valid range [0, 10]
        predictions = np.clip(predictions, 0, 10)
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = {}
            for feature, importance in zip(self.feature_columns, self.model.feature_importances_):
                importance_dict[feature] = importance
            
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def save_model(self, file_path: str = "models/priority_model.pkl") -> None:
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")
    
    def load_model(self, file_path: str = "models/priority_model.pkl") -> None:
        """Load trained model from file"""
        try:
            model_data = joblib.load(file_path)
            
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {file_path}")
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def hyperparameter_tuning(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        logger.info("Performing hyperparameter tuning")
        
        X = self.prepare_features(df)
        
        if target_column and target_column in df.columns:
            y = df[target_column].values
        else:
            y = self.generate_training_labels(df)
        
        # Define parameter grids for different models
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        else:
            logger.info("No hyperparameter tuning available for linear regression")
            return {}
        
        # Grid search
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='r2', n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best CV score: {results['best_score']:.3f}")
        
        return results


def create_sample_priority_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample data for testing the priority model"""
    np.random.seed(42)
    
    # Generate synthetic task data
    data = {
        'urgency_score': np.random.uniform(0, 10, n_samples),
        'complexity_score': np.random.uniform(1, 10, n_samples),
        'days_until_deadline': np.random.randint(0, 365, n_samples),
        'deadline_urgency_multiplier': np.random.uniform(0.5, 2.0, n_samples),
        'estimated_hours': np.random.uniform(0.5, 40, n_samples),
        'category': np.random.choice(['bug', 'feature', 'maintenance', 'security'], n_samples),
        'status': np.random.choice(['pending', 'in_progress', 'completed'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some realistic correlations
    # More urgent tasks tend to have closer deadlines
    urgent_mask = df['urgency_score'] > 7
    df.loc[urgent_mask, 'days_until_deadline'] = np.random.randint(0, 7, urgent_mask.sum())
    
    # Security and bug tasks are more urgent
    security_bug_mask = df['category'].isin(['security', 'bug'])
    df.loc[security_bug_mask, 'urgency_score'] += np.random.uniform(1, 3, security_bug_mask.sum())
    df['urgency_score'] = np.clip(df['urgency_score'], 0, 10)
    
    return df 