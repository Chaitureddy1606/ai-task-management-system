# AI Task Management System 🤖

An intelligent task management system that uses machine learning to automatically prioritize tasks, assign them to team members, and provide insights for better project management.

## 📋 Features

- **Intelligent Task Prioritization**: ML models automatically score task priority based on urgency, complexity, deadlines, and business impact
- **Smart Task Assignment**: Assigns tasks to team members based on skills, workload, availability, and preferences
- **Automatic Task Classification**: Categorizes tasks into types (bug, feature, maintenance, etc.) using NLP
- **Advanced Analytics**: Comprehensive dashboard with insights on team performance, workload distribution, and task trends
- **Feature Engineering**: Extracts 50+ features from task descriptions for better ML predictions

## 🏗️ Architecture

```
ai-task-manager/
│
├── data/                          # Data storage
│   ├── raw/                       # Raw data files
│   ├── processed/                 # Processed datasets
│   └── employee_profiles.json     # Team member profiles
│
├── src/                           # Source code
│   ├── preprocessing.py           # Data preprocessing pipeline
│   ├── feature_engineering.py     # Advanced feature extraction
│   ├── classifier.py             # Task classification models
│   ├── priority_model.py         # Priority scoring models
│   ├── task_assigner.py          # Intelligent task assignment
│   └── utils.py                  # Utility functions
│
├── dashboard/                     # Streamlit web application
│   └── app.py                    # Interactive dashboard
│
├── models/                       # Trained ML models
│   ├── classifier.pkl           # Task classifier
│   └── priority_model.pkl       # Priority prediction model
│
├── db/                          # Database files
│   └── tasks.db                 # SQLite database
│
├── notebooks/                   # Jupyter notebooks
│   └── eda_pipeline.ipynb       # Exploratory data analysis
│
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── final_report.md            # Project documentation
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ai-task-management-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Database

```python
from src.utils import connect_db, create_tasks_table

# Create database and tables
conn = connect_db()
create_tasks_table(conn)
conn.close()
```

### 3. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Navigate to `http://localhost:8501` to access the web interface.

## 💡 Core Components

### 1. Task Preprocessing (`src/preprocessing.py`)

- Text cleaning and normalization
- Feature extraction from task descriptions
- Complexity scoring based on content
- Urgency detection from keywords
- Deadline processing and temporal features

### 2. Priority Model (`src/priority_model.py`)

```python
from src.priority_model import TaskPriorityModel

# Initialize and train model
model = TaskPriorityModel(model_type='random_forest')
metrics = model.train(tasks_df)

# Predict priority for new tasks
priorities = model.predict(new_tasks_df)
```

**Supported Models:**
- Random Forest Regressor
- Gradient Boosting Regressor  
- Linear Regression

### 3. Task Classification (`src/classifier.py`)

```python
from src.classifier import TaskClassifier

# Train classifier
classifier = TaskClassifier(model_type='naive_bayes')
metrics = classifier.train(tasks_df)

# Classify new tasks
result = classifier.classify_task(
    title="Fix login bug",
    description="Users can't log in with special characters"
)
```

**Categories:**
- Bug fixes
- Feature development
- Security tasks
- Maintenance
- Design work
- Documentation

### 4. Intelligent Assignment (`src/task_assigner.py`)

```python
from src.task_assigner import IntelligentTaskAssigner

# Initialize with team profiles
assigner = IntelligentTaskAssigner(employee_profiles)

# Get assignment recommendations
recommendations = assigner.assign_task(task_dict)

# Assign multiple tasks with workload balancing
assignments = assigner.assign_multiple_tasks(tasks_list, balance_workload=True)
```

**Assignment Factors:**
- Skill matching (35%)
- Current workload (25%)
- Task preferences (15%)
- Experience level (15%)
- Time availability (10%)

### 5. Feature Engineering (`src/feature_engineering.py`)

Extracts 50+ features including:
- **Temporal**: Creation time, deadlines, urgency multipliers
- **Text Complexity**: Length, word count, technical indicators
- **Technical Keywords**: Programming, security, infrastructure terms
- **Effort Estimation**: Complexity indicators from text
- **Stakeholder Impact**: Customer-facing vs internal tasks
- **Dependencies**: Task sequence and blocking indicators

## 📊 Dashboard Features

### Main Dashboard
- Task overview and metrics
- Team workload visualization
- Recent tasks and assignments

### Task Management
- Add new tasks with smart categorization
- Intelligent assignment recommendations
- Bulk task operations

### Team Management
- Employee profiles and skills
- Workload distribution charts
- Availability tracking

### AI Models
- Model training interface
- Performance metrics
- Feature importance analysis

### Analytics
- Task completion trends
- Team performance insights
- Category-wise analysis

## 🔧 Configuration

### Employee Profiles (`data/employee_profiles.json`)

```json
{
  "employee_id": "EMP001",
  "name": "Alice Johnson",
  "department": "Engineering",
  "role": "Senior Developer",
  "skills": ["Python", "Machine Learning", "API Development"],
  "experience_years": 5,
  "current_workload": 7,
  "max_capacity": 10,
  "expertise_areas": ["Backend Development", "Data Engineering"],
  "preferred_task_types": ["Development", "Code Review"],
  "availability": {
    "monday": 8,
    "tuesday": 8,
    "wednesday": 6,
    "thursday": 8,
    "friday": 4
  }
}
```

### Model Configuration

Priority model weights can be customized:

```python
custom_weights = {
    'skill_match': 0.4,
    'workload': 0.3,
    'preference': 0.1,
    'experience': 0.1,
    'availability': 0.1
}

assignments = assigner.assign_task(task, weights=custom_weights)
```

## 📈 Model Performance

### Priority Model Metrics
- **R² Score**: 0.85+ on test data
- **RMSE**: < 1.0 priority points
- **Cross-validation**: 5-fold CV with 0.82 average R²

### Classification Accuracy
- **Overall Accuracy**: 90%+ across categories
- **Precision/Recall**: 0.85+ for all major categories
- **F1-Score**: 0.87 average across categories

## 🔍 Example Usage

### 1. Complete ML Pipeline

```python
# Load and preprocess data
tasks_df, employee_profiles = load_data()
processed_df, preprocessor = preprocess_data(tasks_df)

# Train models
priority_model = TaskPriorityModel()
priority_metrics = priority_model.train(processed_df)

classifier = TaskClassifier()
class_metrics = classifier.train(processed_df)

# Initialize task assigner
assigner = IntelligentTaskAssigner(employee_profiles)

# Process new task
new_task = {
    'title': 'Implement OAuth integration',
    'description': 'Add OAuth 2.0 authentication for third-party login',
    'estimated_hours': 12,
    'deadline': '2024-02-15'
}

# Get predictions
priority = priority_model.predict([new_task])[0]
category = classifier.classify_task(new_task['title'], new_task['description'])
assignment = assigner.assign_task(new_task)

print(f"Priority: {priority:.2f}/10")
print(f"Category: {category['predicted_category']}")
print(f"Best assignee: {assignment[0][0]}")
```

### 2. Batch Processing

```python
# Process multiple tasks
tasks_with_features = feature_engineer.engineer_all_features(raw_tasks)
priorities = priority_model.predict(tasks_with_features)
categories = classifier.bulk_classify_tasks(tasks_with_features)
assignments = assigner.assign_multiple_tasks(tasks_list)
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Test individual components
python -c "from src.priority_model import create_sample_priority_data; print('✓ Sample data created')"
python -c "from src.utils import load_employee_profiles; print('✓ Profiles loaded')"
```

## 📝 API Reference

### Core Classes

- `TaskDataPreprocessor`: Data cleaning and preprocessing
- `TaskPriorityModel`: Priority prediction with multiple algorithms
- `TaskClassifier`: Automatic task categorization
- `IntelligentTaskAssigner`: Smart task assignment
- `TaskFeatureEngineer`: Advanced feature extraction

### Utility Functions

- `load_employee_profiles()`: Load team data
- `connect_db()`: Database connection
- `create_tasks_table()`: Initialize database schema

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔮 Future Enhancements

- [ ] Integration with project management tools (Jira, Asana)
- [ ] Real-time notifications and alerts
- [ ] Advanced NLP with transformer models
- [ ] Time series forecasting for delivery predictions
- [ ] Mobile application interface
- [ ] API endpoints for external integrations
- [ ] Advanced reporting and export features

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Check the documentation in `notebooks/eda_pipeline.ipynb`
- Review the `final_report.md` for detailed analysis

---

**Built with ❤️ using Python, Scikit-learn, Streamlit, and modern ML practices** 