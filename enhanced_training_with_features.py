#!/usr/bin/env python3
"""
Enhanced AI Training with Feature-Rich Dataset
- Uses 187 engineered features for superior performance
- Advanced model training with comprehensive features
- Performance comparison with original dataset
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAITrainer:
    """Enhanced AI trainer using feature-rich dataset"""
    
    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.label_encoders = {}
        self.performance_results = {}
        
    def load_prepared_data(self, data_path, info_path):
        """Load the prepared feature-rich dataset"""
        logger.info("Loading prepared feature-rich dataset...")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Load feature information
        with open(info_path, 'r') as f:
            feature_info = json.load(f)
        
        self.feature_columns = feature_info['feature_columns']
        
        print(f"✅ Loaded dataset: {df.shape}")
        print(f"✅ Feature columns: {len(self.feature_columns)}")
        print(f"✅ TF-IDF features: {len([f for f in self.feature_columns if f.startswith('tfidf_')])}")
        
        return df
    
    def train_enhanced_classifier(self, df, target='category'):
        """Train enhanced classifier with 187 features"""
        logger.info(f"Training enhanced classifier for {target}...")
        
        # Prepare features and target
        X = df[self.feature_columns]
        y = df[target]
        
        # Encode target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoders[target] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest with enhanced parameters
        rf_classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        rf_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_classifier, X, y, cv=5)
        cv_accuracy = cv_scores.mean()
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': rf_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"✅ Enhanced Classifier Results:")
        print(f"   Test Accuracy: {accuracy:.3f}")
        print(f"   CV Accuracy: {cv_accuracy:.3f}")
        print(f"   Top 5 Features:")
        for i, row in feature_importance.head().iterrows():
            print(f"     {row['feature']}: {row['importance']:.3f}")
        
        # Store results
        self.models[f'{target}_classifier'] = rf_classifier
        self.performance_results[f'{target}_classifier'] = {
            'test_accuracy': accuracy,
            'cv_accuracy': cv_accuracy,
            'feature_importance': feature_importance
        }
        
        return rf_classifier
    
    def train_enhanced_status_predictor(self, df):
        """Train enhanced status predictor"""
        logger.info("Training enhanced status predictor...")
        
        # Prepare features and target
        X = df[self.feature_columns]
        y = df['status']
        
        # Encode target
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.label_encoders['status'] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost with enhanced parameters
        xgb_classifier = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Train model
        xgb_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = xgb_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(xgb_classifier, X, y, cv=5)
        cv_accuracy = cv_scores.mean()
        
        print(f"✅ Enhanced Status Predictor Results:")
        print(f"   Test Accuracy: {accuracy:.3f}")
        print(f"   CV Accuracy: {cv_accuracy:.3f}")
        
        # Store results
        self.models['status_predictor'] = xgb_classifier
        self.performance_results['status_predictor'] = {
            'test_accuracy': accuracy,
            'cv_accuracy': cv_accuracy
        }
        
        return xgb_classifier
    
    def train_enhanced_priority_predictor(self, df):
        """Train enhanced priority predictor"""
        logger.info("Training enhanced priority predictor...")
        
        # Create priority score (target variable)
        df['priority_score'] = (
            df['urgency_score'] * 0.3 +
            df['complexity_score'] * 0.2 +
            df['business_impact'] * 0.3 +
            df['deadline_urgency'] * 0.2
        )
        
        # Prepare features and target
        X = df[self.feature_columns]
        y = df['priority_score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest Regressor
        rf_regressor = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        rf_regressor.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf_regressor.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_regressor, X, y, cv=5, scoring='r2')
        cv_r2 = cv_scores.mean()
        
        print(f"✅ Enhanced Priority Predictor Results:")
        print(f"   Test R²: {r2:.3f}")
        print(f"   CV R²: {cv_r2:.3f}")
        print(f"   MAE: {mae:.3f}")
        
        # Store results
        self.models['priority_predictor'] = rf_regressor
        self.performance_results['priority_predictor'] = {
            'test_r2': r2,
            'cv_r2': cv_r2,
            'mae': mae
        }
        
        return rf_regressor
    
    def train_enhanced_assigner(self, df):
        """Train enhanced employee assigner"""
        logger.info("Training enhanced employee assigner...")
        
        # Prepare features and target
        X = df[self.feature_columns]
        y = df['assigned_to']
        
        # Encode target
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.label_encoders['assigned_to'] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest Classifier
        rf_assigner = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        rf_assigner.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf_assigner.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_assigner, X, y, cv=5)
        cv_accuracy = cv_scores.mean()
        
        print(f"✅ Enhanced Employee Assigner Results:")
        print(f"   Test Accuracy: {accuracy:.3f}")
        print(f"   CV Accuracy: {cv_accuracy:.3f}")
        
        # Store results
        self.models['employee_assigner'] = rf_assigner
        self.performance_results['employee_assigner'] = {
            'test_accuracy': accuracy,
            'cv_accuracy': cv_accuracy
        }
        
        return rf_assigner
    
    def compare_performance(self, original_results):
        """Compare performance with original models"""
        print("\n📊 Performance Comparison: Enhanced vs Original")
        print("=" * 60)
        
        comparison_data = []
        
        # Compare classification accuracy
        if 'category_classifier' in self.performance_results:
            enhanced_acc = self.performance_results['category_classifier']['test_accuracy']
            original_acc = original_results.get('category_accuracy', 0.725)
            improvement = enhanced_acc - original_acc
            
            comparison_data.append({
                'Model': 'Task Classification',
                'Original': f"{original_acc:.3f}",
                'Enhanced': f"{enhanced_acc:.3f}",
                'Improvement': f"{improvement:+.3f}"
            })
        
        # Compare status prediction
        if 'status_predictor' in self.performance_results:
            enhanced_acc = self.performance_results['status_predictor']['test_accuracy']
            original_acc = original_results.get('status_accuracy', 0.36)
            improvement = enhanced_acc - original_acc
            
            comparison_data.append({
                'Model': 'Status Prediction',
                'Original': f"{original_acc:.3f}",
                'Enhanced': f"{enhanced_acc:.3f}",
                'Improvement': f"{improvement:+.3f}"
            })
        
        # Compare priority prediction
        if 'priority_predictor' in self.performance_results:
            enhanced_r2 = self.performance_results['priority_predictor']['test_r2']
            original_r2 = original_results.get('priority_r2', 0.941)
            improvement = enhanced_r2 - original_r2
            
            comparison_data.append({
                'Model': 'Priority Prediction',
                'Original': f"{original_r2:.3f}",
                'Enhanced': f"{enhanced_r2:.3f}",
                'Improvement': f"{improvement:+.3f}"
            })
        
        # Display comparison table
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            print(df_comparison.to_string(index=False))
        
        return comparison_data
    
    def save_enhanced_models(self, base_path="models/enhanced_features/"):
        """Save enhanced models"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{base_path}{name}.pkl")
        
        # Save label encoders
        joblib.dump(self.label_encoders, f"{base_path}label_encoders.pkl")
        
        # Save feature columns
        with open(f"{base_path}feature_columns.json", 'w') as f:
            json.dump(self.feature_columns, f)
        
        # Save performance results
        with open(f"{base_path}performance_results.json", 'w') as f:
            json.dump(self.performance_results, f, default=str)
        
        print(f"✅ Enhanced models saved to {base_path}")
    
    def train_all_enhanced_models(self, df):
        """Train all enhanced models"""
        logger.info("Training all enhanced models with feature-rich dataset...")
        
        # Train all models
        self.train_enhanced_classifier(df, 'category')
        self.train_enhanced_status_predictor(df)
        self.train_enhanced_priority_predictor(df)
        self.train_enhanced_assigner(df)
        
        print(f"\n🎉 All enhanced models trained successfully!")
        print(f"📊 Total models: {len(self.models)}")
        
        return self.models

def main():
    """Main enhanced training execution"""
    print("🚀 Enhanced AI Training with Feature-Rich Dataset")
    print("=" * 60)
    
    # Initialize trainer
    trainer = EnhancedAITrainer()
    
    # Load prepared data
    data_path = 'data/prepared_training_tasks.csv'
    info_path = 'data/prepared_training_tasks_info.json'
    
    try:
        # Load dataset
        df = trainer.load_prepared_data(data_path, info_path)
        
        # Train all enhanced models
        models = trainer.train_all_enhanced_models(df)
        
        # Compare with original performance
        original_results = {
            'category_accuracy': 0.725,
            'status_accuracy': 0.36,
            'priority_r2': 0.941
        }
        
        comparison = trainer.compare_performance(original_results)
        
        # Save enhanced models
        trainer.save_enhanced_models()
        
        print(f"\n🎉 Enhanced training completed successfully!")
        print(f"📊 Models trained: {len(models)}")
        print(f"📈 Features used: {len(trainer.feature_columns)}")
        
        return trainer
        
    except Exception as e:
        logger.error(f"Error in enhanced training: {e}")
        return None

if __name__ == "__main__":
    main() 