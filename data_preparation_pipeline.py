#!/usr/bin/env python3
"""
Comprehensive Data Preparation Pipeline for AI Task Management
- Clean and feature-rich dataset preparation
- NLP analysis with TF-IDF/BERT embeddings
- Historical outcomes analysis
- Optional enhancements for robust AI
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparationPipeline:
    """Comprehensive data preparation pipeline for AI task management"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_and_clean_data(self, filepath):
        """Load and clean the raw dataset"""
        logger.info(f"Loading data from: {filepath}")
        
        # Load data
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} tasks with {len(df.columns)} columns")
        
        # Display initial data info
        print(f"\n📊 Initial Dataset Info:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def clean_text_features(self, df):
        """Clean and prepare text features (title, description)"""
        logger.info("Cleaning text features...")
        
        # Clean title
        if 'title' in df.columns:
            df['title_clean'] = df['title'].astype(str).apply(self._clean_text)
            print(f"✅ Cleaned {len(df)} titles")
        
        # Clean description
        if 'description' in df.columns:
            df['description_clean'] = df['description'].astype(str).apply(self._clean_text)
            print(f"✅ Cleaned {len(df)} descriptions")
        
        # Combine title and description for NLP analysis
        df['text_combined'] = df['title_clean'] + " " + df['description_clean']
        
        return df
    
    def _clean_text(self, text):
        """Clean text for NLP analysis"""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.\,\!\?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Convert to lowercase
        text = text.lower().strip()
        
        return text
    
    def create_nlp_features(self, df, use_bert=False):
        """Create NLP features using TF-IDF or BERT embeddings"""
        logger.info("Creating NLP features...")
        
        if use_bert:
            # BERT embeddings (placeholder for future implementation)
            logger.info("BERT embeddings not yet implemented, using TF-IDF")
            use_bert = False
        
        if not use_bert:
            # TF-IDF features
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            
            # Fit and transform
            tfidf_features = self.tfidf_vectorizer.fit_transform(df['text_combined'])
            
            # Convert to DataFrame
            tfidf_df = pd.DataFrame(
                tfidf_features.toarray(),
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )
            
            # Add to main dataframe
            df = pd.concat([df, tfidf_df], axis=1)
            
            print(f"✅ Created {tfidf_features.shape[1]} TF-IDF features")
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        logger.info("Encoding categorical features...")
        
        categorical_columns = ['category', 'priority', 'status', 'assigned_to', 'employee_role']
        
        for col in categorical_columns:
            if col in df.columns:
                # Create label encoder
                le = LabelEncoder()
                
                # Handle missing values
                df[col] = df[col].fillna('unknown')
                
                # Fit and transform
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                
                # Store encoder
                self.label_encoders[col] = le
                
                print(f"✅ Encoded {col}: {len(le.classes_)} unique values")
        
        return df
    
    def create_skill_features(self, df):
        """Create skill-based features"""
        logger.info("Creating skill features...")
        
        if 'employee_skills' in df.columns:
            # Parse skills (assuming comma-separated)
            df['skills_list'] = df['employee_skills'].fillna('').str.split(',')
            
            # Create skill count feature
            df['skill_count'] = df['skills_list'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            
            # Create skill match features (placeholder for more sophisticated matching)
            df['has_technical_skills'] = df['employee_skills'].str.contains('python|javascript|java|react|api', case=False).astype(int)
            df['has_testing_skills'] = df['employee_skills'].str.contains('testing|qa|automation', case=False).astype(int)
            df['has_management_skills'] = df['employee_skills'].str.contains('management|planning|coordination', case=False).astype(int)
            
            print(f"✅ Created skill-based features")
        
        return df
    
    def create_temporal_features(self, df):
        """Create temporal and deadline features"""
        logger.info("Creating temporal features...")
        
        # Deadline urgency
        df['deadline_urgency'] = df['days_until_deadline'].apply(
            lambda x: 10 if x < 0 else (9 if x == 0 else (8 if x <= 3 else (6 if x <= 7 else (4 if x <= 30 else 2))))
        )
        
        # Deadline categories
        df['deadline_category'] = pd.cut(
            df['days_until_deadline'],
            bins=[-np.inf, 0, 3, 7, 30, np.inf],
            labels=['overdue', 'urgent', 'soon', 'normal', 'distant']
        )
        
        # Estimated vs actual time (if available)
        if 'estimated_hours' in df.columns and 'time_taken' in df.columns:
            df['time_accuracy'] = df['estimated_hours'] - df['time_taken']
            df['time_overrun'] = (df['time_taken'] > df['estimated_hours']).astype(int)
        
        print(f"✅ Created temporal features")
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between different variables"""
        logger.info("Creating interaction features...")
        
        # Urgency × Complexity
        if 'urgency_score' in df.columns and 'complexity_score' in df.columns:
            df['urgency_complexity'] = df['urgency_score'] * df['complexity_score']
        
        # Business impact × Hours
        if 'business_impact' in df.columns and 'estimated_hours' in df.columns:
            df['business_impact_hours'] = df['business_impact'] * df['estimated_hours']
        
        # Priority × Urgency
        if 'priority_encoded' in df.columns and 'urgency_score' in df.columns:
            df['priority_urgency'] = df['priority_encoded'] * df['urgency_score']
        
        # Deadline × Urgency
        if 'deadline_urgency' in df.columns and 'urgency_score' in df.columns:
            df['deadline_urgency_interaction'] = df['deadline_urgency'] * df['urgency_score']
        
        print(f"✅ Created interaction features")
        
        return df
    
    def create_historical_outcome_features(self, df):
        """Create features based on historical outcomes"""
        logger.info("Creating historical outcome features...")
        
        # Success rate by employee (if historical data available)
        if 'assigned_to' in df.columns and 'status' in df.columns:
            employee_success = df[df['status'] == 'completed'].groupby('assigned_to').size()
            employee_total = df.groupby('assigned_to').size()
            employee_success_rate = (employee_success / employee_total).fillna(0)
            
            df['employee_success_rate'] = df['assigned_to'].map(employee_success_rate)
        
        # Category completion rate
        if 'category' in df.columns and 'status' in df.columns:
            category_success = df[df['status'] == 'completed'].groupby('category').size()
            category_total = df.groupby('category').size()
            category_success_rate = (category_success / category_total).fillna(0)
            
            df['category_success_rate'] = df['category'].map(category_success_rate)
        
        # Priority completion rate
        if 'priority' in df.columns and 'status' in df.columns:
            priority_success = df[df['status'] == 'completed'].groupby('priority').size()
            priority_total = df.groupby('priority').size()
            priority_success_rate = (priority_success / priority_total).fillna(0)
            
            df['priority_success_rate'] = df['priority'].map(priority_success_rate)
        
        print(f"✅ Created historical outcome features")
        
        return df
    
    def create_optional_enhancements(self, df):
        """Create optional enhancement features"""
        logger.info("Creating optional enhancement features...")
        
        # Task context features
        df['has_client_impact'] = df['text_combined'].str.contains('client|customer|user', case=False).astype(int)
        df['has_team_dependency'] = df['text_combined'].str.contains('team|collaboration|dependency', case=False).astype(int)
        df['has_security_concern'] = df['text_combined'].str.contains('security|vulnerability|breach', case=False).astype(int)
        df['has_performance_issue'] = df['text_combined'].str.contains('performance|slow|optimization', case=False).astype(int)
        
        # Text length features
        df['title_length'] = df['title_clean'].str.len()
        df['description_length'] = df['description_clean'].str.len()
        df['text_complexity'] = df['text_combined'].str.split().str.len()
        
        # Word frequency features
        df['has_bug_keywords'] = df['text_combined'].str.contains('bug|error|fix|issue', case=False).astype(int)
        df['has_feature_keywords'] = df['text_combined'].str.contains('feature|implement|add|new', case=False).astype(int)
        df['has_testing_keywords'] = df['text_combined'].str.contains('test|testing|qa|verify', case=False).astype(int)
        df['has_documentation_keywords'] = df['text_combined'].str.contains('document|doc|write|create', case=False).astype(int)
        
        print(f"✅ Created optional enhancement features")
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        # Numeric columns - fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"   Filled {col} with median: {median_val}")
        
        # Categorical columns - fill with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'unknown'
                df[col].fillna(mode_val, inplace=True)
                print(f"   Filled {col} with mode: {mode_val}")
        
        return df
    
    def scale_numeric_features(self, df):
        """Scale numeric features"""
        logger.info("Scaling numeric features...")
        
        # Select numeric features (excluding encoded categoricals and TF-IDF features)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        exclude_columns = [col for col in numeric_columns if 'encoded' in col or 'tfidf_' in col]
        scale_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        if len(scale_columns) > 0:
            # Scale features
            df[scale_columns] = self.scaler.fit_transform(df[scale_columns])
            print(f"✅ Scaled {len(scale_columns)} numeric features")
        
        return df
    
    def create_final_feature_set(self, df):
        """Create final feature set for AI training"""
        logger.info("Creating final feature set...")
        
        # Define feature columns
        self.feature_columns = []
        
        # Core features
        core_features = [
            'urgency_score', 'complexity_score', 'days_until_deadline',
            'business_impact', 'estimated_hours', 'deadline_urgency',
            'skill_count', 'has_technical_skills', 'has_testing_skills', 'has_management_skills',
            'urgency_complexity', 'business_impact_hours', 'priority_urgency',
            'deadline_urgency_interaction', 'employee_success_rate', 'category_success_rate',
            'priority_success_rate', 'has_client_impact', 'has_team_dependency',
            'has_security_concern', 'has_performance_issue', 'title_length',
            'description_length', 'text_complexity', 'has_bug_keywords',
            'has_feature_keywords', 'has_testing_keywords', 'has_documentation_keywords'
        ]
        
        # Add available core features
        for feature in core_features:
            if feature in df.columns:
                self.feature_columns.append(feature)
        
        # Add encoded categorical features
        for col in df.columns:
            if col.endswith('_encoded'):
                self.feature_columns.append(col)
        
        # Add TF-IDF features
        tfidf_features = [col for col in df.columns if col.startswith('tfidf_')]
        self.feature_columns.extend(tfidf_features)
        
        print(f"✅ Final feature set: {len(self.feature_columns)} features")
        print(f"   Core features: {len([f for f in self.feature_columns if not f.startswith('tfidf_') and not f.endswith('_encoded')])}")
        print(f"   Encoded features: {len([f for f in self.feature_columns if f.endswith('_encoded')])}")
        print(f"   TF-IDF features: {len([f for f in self.feature_columns if f.startswith('tfidf_')])}")
        
        return df
    
    def prepare_dataset(self, filepath, use_bert=False):
        """Complete data preparation pipeline"""
        logger.info("Starting comprehensive data preparation pipeline...")
        
        # Step 1: Load and clean data
        df = self.load_and_clean_data(filepath)
        
        # Step 2: Clean text features
        df = self.clean_text_features(df)
        
        # Step 3: Create NLP features
        df = self.create_nlp_features(df, use_bert)
        
        # Step 4: Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Step 5: Create skill features
        df = self.create_skill_features(df)
        
        # Step 6: Create temporal features
        df = self.create_temporal_features(df)
        
        # Step 7: Create interaction features
        df = self.create_interaction_features(df)
        
        # Step 8: Create historical outcome features
        df = self.create_historical_outcome_features(df)
        
        # Step 9: Create optional enhancements
        df = self.create_optional_enhancements(df)
        
        # Step 10: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 11: Scale numeric features
        df = self.scale_numeric_features(df)
        
        # Step 12: Create final feature set
        df = self.create_final_feature_set(df)
        
        # Display final dataset info
        print(f"\n📊 Final Dataset Info:")
        print(f"   Shape: {df.shape}")
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def save_prepared_data(self, df, output_path):
        """Save prepared dataset"""
        logger.info(f"Saving prepared dataset to: {output_path}")
        
        # Save main dataset
        df.to_csv(output_path, index=False)
        
        # Save feature information
        feature_info = {
            'feature_columns': self.feature_columns,
            'label_encoders': {k: list(v.classes_) for k, v in self.label_encoders.items()},
            'tfidf_features': self.tfidf_vectorizer.get_feature_names_out().tolist() if self.tfidf_vectorizer else [],
            'preparation_timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(output_path.replace('.csv', '_info.json'), 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"✅ Saved prepared dataset and feature information")
        
        return df

def main():
    """Main data preparation execution"""
    print("🔹 Step 2: Prepare a Clean, Feature-Rich Dataset")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = DataPreparationPipeline()
    
    # Prepare dataset
    input_file = 'data/combined_training_tasks.csv'
    output_file = 'data/prepared_training_tasks.csv'
    
    try:
        # Run complete pipeline
        df = pipeline.prepare_dataset(input_file, use_bert=False)
        
        # Save prepared data
        pipeline.save_prepared_data(df, output_file)
        
        print(f"\n🎉 Data preparation completed successfully!")
        print(f"📁 Input: {input_file}")
        print(f"📁 Output: {output_file}")
        print(f"📊 Features: {len(pipeline.feature_columns)}")
        
        # Display sample of prepared features
        print(f"\n📋 Sample of prepared features:")
        feature_sample = pipeline.feature_columns[:10]
        print(f"   {feature_sample}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in data preparation: {e}")
        return None

if __name__ == "__main__":
    main() 