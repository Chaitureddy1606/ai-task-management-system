"""
Feature Engineering Module for AI Task Management System
Advanced feature extraction and engineering for task analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class TaskFeatureEngineer:
    """Advanced feature engineering for task management data"""
    
    def __init__(self):
        self.feature_cache = {}
        self.patterns_cache = {}
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from task data"""
        df_features = df.copy()
        
        # Convert date columns
        date_columns = ['created_at', 'deadline', 'updated_at']
        for col in date_columns:
            if col in df_features.columns:
                df_features[col] = pd.to_datetime(df_features[col], errors='coerce')
        
        if 'created_at' in df_features.columns:
            # Extract creation time features
            df_features['created_hour'] = df_features['created_at'].dt.hour
            df_features['created_day_of_week'] = df_features['created_at'].dt.dayofweek
            df_features['created_month'] = df_features['created_at'].dt.month
            df_features['created_quarter'] = df_features['created_at'].dt.quarter
            
            # Time since creation
            now = pd.Timestamp.now()
            df_features['days_since_creation'] = (now - df_features['created_at']).dt.days
        
        if 'deadline' in df_features.columns:
            # Deadline features
            now = pd.Timestamp.now()
            df_features['days_until_deadline'] = (df_features['deadline'] - now).dt.days
            df_features['is_overdue'] = (df_features['deadline'] < now).astype(int)
            
            # Deadline pressure categories
            df_features['deadline_pressure'] = pd.cut(
                df_features['days_until_deadline'],
                bins=[-float('inf'), 0, 1, 7, 30, float('inf')],
                labels=['overdue', 'today', 'this_week', 'this_month', 'future']
            )
        
        # Time between creation and deadline (task timeline)
        if 'created_at' in df_features.columns and 'deadline' in df_features.columns:
            df_features['task_timeline_days'] = (
                df_features['deadline'] - df_features['created_at']
            ).dt.days
            
            # Timeline categories
            df_features['timeline_category'] = pd.cut(
                df_features['task_timeline_days'],
                bins=[0, 1, 7, 30, 90, float('inf')],
                labels=['same_day', 'week', 'month', 'quarter', 'long_term']
            )
        
        return df_features
    
    def extract_text_complexity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract complexity features from text content"""
        df_features = df.copy()
        
        for text_col in ['title', 'description']:
            if text_col in df_features.columns:
                col_prefix = text_col
                
                # Basic text statistics
                df_features[f'{col_prefix}_length'] = df_features[text_col].str.len().fillna(0)
                df_features[f'{col_prefix}_word_count'] = df_features[text_col].str.split().str.len().fillna(0)
                df_features[f'{col_prefix}_sentence_count'] = df_features[text_col].str.count(r'[.!?]').fillna(0)
                
                # Average word length
                df_features[f'{col_prefix}_avg_word_length'] = df_features.apply(
                    lambda row: np.mean([len(word) for word in str(row[text_col]).split()]) 
                    if pd.notna(row[text_col]) and row[text_col] else 0, axis=1
                )
                
                # Technical complexity indicators
                df_features[f'{col_prefix}_has_code'] = df_features[text_col].str.contains(
                    r'[\{\[\(\)]|def |class |import |SELECT|UPDATE|DELETE|CREATE', 
                    case=False, na=False
                ).astype(int)
                
                # URL/link indicators
                df_features[f'{col_prefix}_has_url'] = df_features[text_col].str.contains(
                    r'http[s]?://|www\.|\.com|\.org', case=False, na=False
                ).astype(int)
                
                # Question indicators (complexity often correlates with questions)
                df_features[f'{col_prefix}_question_count'] = df_features[text_col].str.count(r'\?').fillna(0)
                
                # Uppercase ratio (might indicate urgency)
                df_features[f'{col_prefix}_uppercase_ratio'] = df_features.apply(
                    lambda row: sum(1 for c in str(row[text_col]) if c.isupper()) / len(str(row[text_col])) 
                    if pd.notna(row[text_col]) and len(str(row[text_col])) > 0 else 0, axis=1
                )
        
        return df_features
    
    def extract_technical_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract technical keyword features"""
        df_features = df.copy()
        
        # Technical keyword categories
        technical_keywords = {
            'programming': [
                'api', 'database', 'sql', 'python', 'javascript', 'react', 'node',
                'framework', 'library', 'algorithm', 'function', 'method', 'class'
            ],
            'infrastructure': [
                'server', 'cloud', 'aws', 'docker', 'kubernetes', 'deployment',
                'infrastructure', 'nginx', 'apache', 'linux', 'windows'
            ],
            'security': [
                'security', 'authentication', 'authorization', 'encryption',
                'ssl', 'certificate', 'vulnerability', 'audit', 'compliance'
            ],
            'data': [
                'data', 'analytics', 'report', 'dashboard', 'visualization',
                'machine learning', 'ai', 'model', 'prediction', 'analysis'
            ],
            'testing': [
                'test', 'testing', 'unit test', 'integration', 'qa', 'quality',
                'bug', 'issue', 'defect', 'validation', 'verification'
            ]
        }
        
        # Combine title and description for keyword extraction
        text_content = (
            df_features['title'].fillna('') + ' ' + 
            df_features['description'].fillna('')
        ).str.lower()
        
        # Count keywords for each category
        for category, keywords in technical_keywords.items():
            df_features[f'keyword_{category}_count'] = text_content.apply(
                lambda text: sum(1 for keyword in keywords if keyword in text)
            )
            
            # Binary indicators
            df_features[f'has_{category}_keywords'] = (
                df_features[f'keyword_{category}_count'] > 0
            ).astype(int)
        
        # Overall technical complexity score
        keyword_columns = [f'keyword_{cat}_count' for cat in technical_keywords.keys()]
        df_features['technical_complexity_score'] = df_features[keyword_columns].sum(axis=1)
        
        return df_features
    
    def extract_urgency_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract urgency and priority indicators from text"""
        df_features = df.copy()
        
        # Urgency keywords
        urgency_patterns = {
            'high_urgency': [
                'urgent', 'asap', 'immediately', 'critical', 'emergency',
                'high priority', 'important', 'deadline', 'rush'
            ],
            'medium_urgency': [
                'soon', 'quick', 'fast', 'priority', 'needed', 'required'
            ],
            'low_urgency': [
                'when possible', 'eventually', 'nice to have', 'low priority',
                'future', 'someday', 'optional'
            ]
        }
        
        # Combine text for analysis
        text_content = (
            df_features['title'].fillna('') + ' ' + 
            df_features['description'].fillna('')
        ).str.lower()
        
        # Extract urgency scores
        for urgency_level, keywords in urgency_patterns.items():
            df_features[f'{urgency_level}_indicators'] = text_content.apply(
                lambda text: sum(1 for keyword in keywords if keyword in text)
            )
        
        # Calculate overall urgency score
        df_features['text_urgency_score'] = (
            df_features['high_urgency_indicators'] * 3 +
            df_features['medium_urgency_indicators'] * 2 +
            df_features['low_urgency_indicators'] * (-1)  # Negative for low urgency
        )
        
        # Deadline urgency (if deadline exists)
        if 'days_until_deadline' in df_features.columns:
            df_features['deadline_urgency_score'] = df_features['days_until_deadline'].apply(
                lambda days: 10 if days < 0 else  # Overdue
                             8 if days <= 1 else   # Due today/tomorrow
                             6 if days <= 7 else   # This week
                             4 if days <= 30 else  # This month
                             2                     # Future
            )
        
        return df_features
    
    def extract_effort_estimation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features related to effort estimation"""
        df_features = df.copy()
        
        # Effort indicators from text
        effort_keywords = {
            'high_effort': [
                'implement', 'develop', 'create', 'build', 'design', 'architecture',
                'complex', 'integrate', 'migration', 'refactor', 'optimization'
            ],
            'medium_effort': [
                'update', 'modify', 'enhance', 'improve', 'add', 'configure',
                'setup', 'install', 'test', 'review'
            ],
            'low_effort': [
                'fix', 'change', 'adjust', 'quick', 'simple', 'small',
                'minor', 'tweak', 'correct'
            ]
        }
        
        # Combine text
        text_content = (
            df_features['title'].fillna('') + ' ' + 
            df_features['description'].fillna('')
        ).str.lower()
        
        # Extract effort indicators
        for effort_level, keywords in effort_keywords.items():
            df_features[f'{effort_level}_indicators'] = text_content.apply(
                lambda text: sum(1 for keyword in keywords if keyword in text)
            )
        
        # Calculate text-based effort score
        df_features['text_effort_score'] = (
            df_features['high_effort_indicators'] * 3 +
            df_features['medium_effort_indicators'] * 2 +
            df_features['low_effort_indicators'] * 1
        )
        
        # Combine with estimated hours if available
        if 'estimated_hours' in df_features.columns:
            # Normalize estimated hours to 1-10 scale
            max_hours = df_features['estimated_hours'].quantile(0.95)
            df_features['normalized_effort_hours'] = np.clip(
                df_features['estimated_hours'] / max_hours * 10, 1, 10
            )
            
            # Combined effort score
            df_features['combined_effort_score'] = (
                df_features['text_effort_score'] * 0.3 +
                df_features['normalized_effort_hours'] * 0.7
            )
        else:
            df_features['combined_effort_score'] = df_features['text_effort_score']
        
        return df_features
    
    def extract_dependency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features indicating task dependencies"""
        df_features = df.copy()
        
        # Dependency indicators
        dependency_keywords = [
            'depends on', 'after', 'before', 'requires', 'needs', 'prerequisite',
            'blocked by', 'waiting for', 'once', 'when', 'if completed'
        ]
        
        # Combine text
        text_content = (
            df_features['title'].fillna('') + ' ' + 
            df_features['description'].fillna('')
        ).str.lower()
        
        # Count dependency indicators
        df_features['dependency_indicators'] = text_content.apply(
            lambda text: sum(1 for keyword in dependency_keywords if keyword in text)
        )
        
        df_features['has_dependencies'] = (df_features['dependency_indicators'] > 0).astype(int)
        
        # Task sequence indicators
        sequence_patterns = [
            r'step \d+', r'phase \d+', r'part \d+', r'stage \d+',
            r'first', r'second', r'third', r'final', r'last'
        ]
        
        df_features['sequence_indicators'] = text_content.apply(
            lambda text: sum(1 for pattern in sequence_patterns if re.search(pattern, text))
        )
        
        return df_features
    
    def extract_stakeholder_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features related to stakeholders and impact"""
        df_features = df.copy()
        
        # Stakeholder keywords
        stakeholder_keywords = {
            'customer_facing': [
                'customer', 'user', 'client', 'public', 'external', 'frontend',
                'ui', 'ux', 'interface', 'dashboard', 'website'
            ],
            'internal': [
                'internal', 'admin', 'backend', 'database', 'server',
                'infrastructure', 'maintenance', 'cleanup', 'optimization'
            ],
            'management': [
                'report', 'metrics', 'analytics', 'dashboard', 'kpi',
                'management', 'executive', 'director', 'stakeholder'
            ]
        }
        
        # Combine text
        text_content = (
            df_features['title'].fillna('') + ' ' + 
            df_features['description'].fillna('')
        ).str.lower()
        
        # Extract stakeholder impact scores
        for stakeholder_type, keywords in stakeholder_keywords.items():
            df_features[f'{stakeholder_type}_impact'] = text_content.apply(
                lambda text: sum(1 for keyword in keywords if keyword in text)
            )
        
        # Overall business impact score
        df_features['business_impact_score'] = (
            df_features['customer_facing_impact'] * 3 +
            df_features['management_impact'] * 2 +
            df_features['internal_impact'] * 1
        )
        
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different aspects"""
        df_features = df.copy()
        
        # Effort vs Urgency interaction
        if 'combined_effort_score' in df_features.columns and 'text_urgency_score' in df_features.columns:
            df_features['effort_urgency_ratio'] = df_features['text_urgency_score'] / (df_features['combined_effort_score'] + 1)
            df_features['quick_win_score'] = (df_features['text_urgency_score'] * 2) / (df_features['combined_effort_score'] + 1)
        
        # Technical complexity vs business impact
        if 'technical_complexity_score' in df_features.columns and 'business_impact_score' in df_features.columns:
            df_features['tech_business_ratio'] = df_features['business_impact_score'] / (df_features['technical_complexity_score'] + 1)
        
        # Deadline pressure vs effort
        if 'deadline_urgency_score' in df_features.columns and 'combined_effort_score' in df_features.columns:
            df_features['deadline_effort_pressure'] = df_features['deadline_urgency_score'] / (df_features['combined_effort_score'] + 1)
        
        return df_features
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        logger.info("Starting comprehensive feature engineering")
        
        # Start with original dataframe
        df_engineered = df.copy()
        
        # Apply all feature engineering steps
        df_engineered = self.extract_temporal_features(df_engineered)
        df_engineered = self.extract_text_complexity_features(df_engineered)
        df_engineered = self.extract_technical_keywords(df_engineered)
        df_engineered = self.extract_urgency_indicators(df_engineered)
        df_engineered = self.extract_effort_estimation_features(df_engineered)
        df_engineered = self.extract_dependency_features(df_engineered)
        df_engineered = self.extract_stakeholder_features(df_engineered)
        df_engineered = self.create_interaction_features(df_engineered)
        
        # Fill any NaN values created during feature engineering
        numeric_columns = df_engineered.select_dtypes(include=[np.number]).columns
        df_engineered[numeric_columns] = df_engineered[numeric_columns].fillna(0)
        
        logger.info(f"Feature engineering completed. Created {len(df_engineered.columns) - len(df.columns)} new features")
        
        return df_engineered
    
    def get_feature_importance_ranking(self, df: pd.DataFrame, target_column: str = 'priority_score') -> List[Tuple[str, float]]:
        """Calculate feature importance using correlation with target"""
        
        if target_column not in df.columns:
            logger.warning(f"Target column {target_column} not found")
            return []
        
        # Calculate correlations
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_columns].corr()[target_column].abs().sort_values(ascending=False)
        
        # Remove self-correlation
        correlations = correlations.drop(target_column, errors='ignore')
        
        # Return as list of tuples
        return [(feature, corr) for feature, corr in correlations.items()]
    
    def create_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary of engineered features"""
        
        feature_groups = {
            'temporal': [col for col in df.columns if any(x in col for x in ['created', 'deadline', 'days', 'timeline'])],
            'text_complexity': [col for col in df.columns if any(x in col for x in ['length', 'word_count', 'sentence'])],
            'technical': [col for col in df.columns if 'keyword' in col or 'technical' in col],
            'urgency': [col for col in df.columns if 'urgency' in col],
            'effort': [col for col in df.columns if 'effort' in col],
            'stakeholder': [col for col in df.columns if any(x in col for x in ['impact', 'stakeholder', 'customer', 'internal'])],
            'interaction': [col for col in df.columns if any(x in col for x in ['ratio', 'pressure', 'win'])]
        }
        
        summary = {
            'total_features': len(df.columns),
            'feature_groups': {group: len(features) for group, features in feature_groups.items()},
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return summary


def create_feature_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, TaskFeatureEngineer]:
    """Create complete feature engineering pipeline"""
    
    engineer = TaskFeatureEngineer()
    df_features = engineer.engineer_all_features(df)
    
    return df_features, engineer 