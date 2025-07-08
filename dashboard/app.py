"""
Streamlit Dashboard for AI Task Management System
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import json

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.utils import load_employee_profiles, connect_db, create_tasks_table
    from src.preprocessing import TaskDataPreprocessor
    from src.priority_model import TaskPriorityModel, create_sample_priority_data
    from src.task_assigner import IntelligentTaskAssigner
except ImportError:
    st.error("Could not import required modules. Please ensure all dependencies are installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Task Management System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_database():
    """Initialize database with sample data if needed"""
    try:
        conn = connect_db()
        create_tasks_table(conn)
        
        # Check if we have any tasks
        cursor = conn.execute("SELECT COUNT(*) FROM tasks")
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Create sample tasks
            sample_tasks = [
                {
                    'title': 'Implement user authentication',
                    'description': 'Develop secure user login and registration system with JWT tokens',
                    'category': 'security',
                    'estimated_hours': 16,
                    'deadline': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                    'status': 'pending'
                },
                {
                    'title': 'Fix database performance issue',
                    'description': 'Optimize slow running queries in the user management module',
                    'category': 'bug',
                    'estimated_hours': 8,
                    'deadline': (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
                    'status': 'pending'
                },
                {
                    'title': 'Design new dashboard layout',
                    'description': 'Create modern responsive dashboard design with improved UX',
                    'category': 'design',
                    'estimated_hours': 20,
                    'deadline': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),
                    'status': 'pending'
                }
            ]
            
            for task in sample_tasks:
                conn.execute("""
                    INSERT INTO tasks (title, description, category, estimated_hours, deadline, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (task['title'], task['description'], task['category'], 
                     task['estimated_hours'], task['deadline'], task['status']))
            
            conn.commit()
            st.success("Sample tasks created!")
        
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Database initialization error: {e}")
        return False

def load_data():
    """Load task data and employee profiles"""
    try:
        # Load tasks from database
        conn = connect_db()
        tasks_df = pd.read_sql_query("SELECT * FROM tasks", conn)
        conn.close()
        
        # Load employee profiles
        employee_profiles = load_employee_profiles()
        
        return tasks_df, employee_profiles
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), []

def preprocess_data(tasks_df):
    """Preprocess task data"""
    if tasks_df.empty:
        return tasks_df
    
    try:
        preprocessor = TaskDataPreprocessor()
        processed_df = preprocessor.fit_transform(tasks_df)
        return processed_df, preprocessor
    
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return tasks_df, None

def train_priority_model(processed_df):
    """Train priority prediction model"""
    if processed_df.empty:
        return None
    
    try:
        model = TaskPriorityModel()
        metrics = model.train(processed_df)
        return model, metrics
    
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, {}

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Task Management System</h1>', unsafe_allow_html=True)
    
    # Initialize database
    if not initialize_database():
        st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Tasks", "Team Management", "AI Models", "Analytics"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        tasks_df, employee_profiles = load_data()
    
    if page == "Dashboard":
        show_dashboard(tasks_df, employee_profiles)
    elif page == "Tasks":
        show_tasks_page(tasks_df, employee_profiles)
    elif page == "Team Management":
        show_team_page(employee_profiles)
    elif page == "AI Models":
        show_ai_models_page(tasks_df)
    elif page == "Analytics":
        show_analytics_page(tasks_df, employee_profiles)

def show_dashboard(tasks_df, employee_profiles):
    """Show main dashboard"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_tasks = len(tasks_df)
        st.metric("Total Tasks", total_tasks)
    
    with col2:
        pending_tasks = len(tasks_df[tasks_df['status'] == 'pending']) if not tasks_df.empty else 0
        st.metric("Pending Tasks", pending_tasks)
    
    with col3:
        total_employees = len(employee_profiles)
        st.metric("Team Members", total_employees)
    
    with col4:
        if not tasks_df.empty and 'estimated_hours' in tasks_df.columns:
            avg_hours = tasks_df['estimated_hours'].mean()
            st.metric("Avg. Task Hours", f"{avg_hours:.1f}")
        else:
            st.metric("Avg. Task Hours", "N/A")
    
    # Charts
    if not tasks_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tasks by Status")
            status_counts = tasks_df['status'].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Task Distribution by Status"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Tasks by Category")
            if 'category' in tasks_df.columns:
                category_counts = tasks_df['category'].value_counts()
                fig = px.bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    title="Tasks by Category"
                )
                fig.update_layout(
                    xaxis_title="Category",
                    yaxis_title="Number of Tasks"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Recent tasks
    st.subheader("Recent Tasks")
    if not tasks_df.empty:
        recent_tasks = tasks_df.head(5)[['title', 'category', 'status', 'deadline']]
        st.dataframe(recent_tasks, use_container_width=True)
    else:
        st.info("No tasks found. Add some tasks to get started!")

def show_tasks_page(tasks_df, employee_profiles):
    """Show tasks management page"""
    
    st.header("Task Management")
    
    # Task creation form
    with st.expander("Add New Task", expanded=False):
        with st.form("new_task_form"):
            title = st.text_input("Task Title")
            description = st.text_area("Description")
            category = st.selectbox(
                "Category",
                ["bug", "feature", "maintenance", "security", "design", "documentation"]
            )
            estimated_hours = st.number_input("Estimated Hours", min_value=0.5, value=8.0, step=0.5)
            deadline = st.date_input("Deadline", value=datetime.now() + timedelta(days=7))
            
            submitted = st.form_submit_button("Add Task")
            
            if submitted and title:
                try:
                    conn = connect_db()
                    conn.execute("""
                        INSERT INTO tasks (title, description, category, estimated_hours, deadline, status)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (title, description, category, estimated_hours, deadline.strftime('%Y-%m-%d'), 'pending'))
                    conn.commit()
                    conn.close()
                    st.success("Task added successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding task: {e}")
    
    # Task assignment
    st.subheader("Intelligent Task Assignment")
    
    if not tasks_df.empty and employee_profiles:
        pending_tasks = tasks_df[tasks_df['status'] == 'pending']
        
        if not pending_tasks.empty:
            # Preprocess data for assignment
            try:
                processed_df, preprocessor = preprocess_data(tasks_df)
                
                # Initialize task assigner
                assigner = IntelligentTaskAssigner(employee_profiles)
                
                # Select task to assign
                task_options = {f"{row['title']} (ID: {row['id']})": row 
                              for _, row in pending_tasks.iterrows()}
                
                selected_task_key = st.selectbox("Select task to assign:", list(task_options.keys()))
                
                if selected_task_key:
                    selected_task = task_options[selected_task_key]
                    
                    # Get assignment recommendations
                    recommendations = assigner.assign_task(selected_task.to_dict())
                    
                    if recommendations:
                        st.write("**Assignment Recommendations:**")
                        
                        rec_df = pd.DataFrame(recommendations, columns=['Employee ID', 'Score'])
                        rec_df['Score'] = rec_df['Score'].round(3)
                        
                        # Add employee names
                        emp_names = {}
                        for emp in employee_profiles:
                            emp_names[emp['employee_id']] = emp.get('name', 'Unknown')
                        
                        rec_df['Employee Name'] = rec_df['Employee ID'].map(emp_names)
                        rec_df = rec_df[['Employee ID', 'Employee Name', 'Score']]
                        
                        st.dataframe(rec_df, use_container_width=True)
                        
                        # Assign button
                        if st.button("Assign to Top Candidate"):
                            best_employee = recommendations[0][0]
                            try:
                                conn = connect_db()
                                conn.execute(
                                    "UPDATE tasks SET assigned_to = ?, status = 'assigned' WHERE id = ?",
                                    (best_employee, selected_task['id'])
                                )
                                conn.commit()
                                conn.close()
                                st.success(f"Task assigned to {emp_names.get(best_employee, best_employee)}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error assigning task: {e}")
                
            except Exception as e:
                st.error(f"Error in task assignment: {e}")
        else:
            st.info("No pending tasks available for assignment.")
    
    # Tasks table
    st.subheader("All Tasks")
    if not tasks_df.empty:
        # Add employee names to tasks
        if 'assigned_to' in tasks_df.columns:
            emp_names = {emp['employee_id']: emp.get('name', 'Unknown') for emp in employee_profiles}
            tasks_display = tasks_df.copy()
            tasks_display['assigned_to_name'] = tasks_display['assigned_to'].map(emp_names).fillna('Unassigned')
        else:
            tasks_display = tasks_df.copy()
            tasks_display['assigned_to_name'] = 'Unassigned'
        
        # Display tasks
        display_columns = ['title', 'category', 'status', 'assigned_to_name', 'deadline', 'estimated_hours']
        available_columns = [col for col in display_columns if col in tasks_display.columns]
        
        st.dataframe(
            tasks_display[available_columns],
            use_container_width=True
        )

def show_team_page(employee_profiles):
    """Show team management page"""
    
    st.header("Team Management")
    
    if not employee_profiles:
        st.warning("No employee profiles found. Please check the employee_profiles.json file.")
        return
    
    # Team overview
    st.subheader("Team Overview")
    
    # Create team summary
    team_data = []
    for emp in employee_profiles:
        team_data.append({
            'ID': emp.get('employee_id', ''),
            'Name': emp.get('name', 'Unknown'),
            'Department': emp.get('department', 'Unknown'),
            'Role': emp.get('role', 'Unknown'),
            'Experience (Years)': emp.get('experience_years', 0),
            'Current Workload': emp.get('current_workload', 0),
            'Max Capacity': emp.get('max_capacity', 10),
            'Utilization %': round((emp.get('current_workload', 0) / emp.get('max_capacity', 10)) * 100, 1)
        })
    
    team_df = pd.DataFrame(team_data)
    st.dataframe(team_df, use_container_width=True)
    
    # Workload visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Workload Distribution")
        fig = px.bar(
            team_df,
            x='Name',
            y=['Current Workload', 'Max Capacity'],
            barmode='group',
            title="Current Workload vs Capacity"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Team Utilization")
        fig = px.bar(
            team_df,
            x='Name',
            y='Utilization %',
            title="Team Member Utilization",
            color='Utilization %',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Employee details
    st.subheader("Employee Details")
    selected_emp = st.selectbox(
        "Select employee:",
        options=[emp['name'] for emp in employee_profiles]
    )
    
    if selected_emp:
        emp_data = next(emp for emp in employee_profiles if emp['name'] == selected_emp)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Name:** {emp_data.get('name', 'Unknown')}")
            st.write(f"**Department:** {emp_data.get('department', 'Unknown')}")
            st.write(f"**Role:** {emp_data.get('role', 'Unknown')}")
            st.write(f"**Experience:** {emp_data.get('experience_years', 0)} years")
        
        with col2:
            st.write(f"**Skills:** {', '.join(emp_data.get('skills', []))}")
            st.write(f"**Expertise:** {', '.join(emp_data.get('expertise_areas', []))}")
            st.write(f"**Preferred Tasks:** {', '.join(emp_data.get('preferred_task_types', []))}")
        
        # Availability chart
        availability = emp_data.get('availability', {})
        if availability:
            st.subheader("Weekly Availability")
            days = list(availability.keys())
            hours = list(availability.values())
            
            fig = px.bar(x=days, y=hours, title="Daily Availability (Hours)")
            st.plotly_chart(fig, use_container_width=True)

def show_ai_models_page(tasks_df):
    """Show AI models page"""
    
    st.header("AI Models")
    
    if tasks_df.empty:
        st.warning("No task data available for model training.")
        return
    
    # Model training section
    st.subheader("Priority Prediction Model")
    
    if st.button("Train Priority Model"):
        with st.spinner("Training model..."):
            try:
                # Preprocess data
                processed_df, preprocessor = preprocess_data(tasks_df)
                
                # Train model
                model, metrics = train_priority_model(processed_df)
                
                if model and metrics:
                    st.success("Model trained successfully!")
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("R² Score", f"{metrics.get('r2', 0):.3f}")
                    with col2:
                        st.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
                    with col3:
                        st.metric("MAE", f"{metrics.get('mae', 0):.3f}")
                    
                    # Feature importance
                    importance = model.get_feature_importance()
                    if importance:
                        st.subheader("Feature Importance")
                        
                        features = list(importance.keys())[:10]  # Top 10
                        importances = list(importance.values())[:10]
                        
                        fig = px.bar(
                            x=importances,
                            y=features,
                            orientation='h',
                            title="Top 10 Most Important Features"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Save model
                    model.save_model()
                    st.info("Model saved to models/priority_model.pkl")
                
            except Exception as e:
                st.error(f"Error training model: {e}")
    
    # Model testing section
    st.subheader("Test Priority Prediction")
    
    with st.form("prediction_form"):
        task_desc = st.text_area("Task Description")
        task_category = st.selectbox("Category", ["bug", "feature", "maintenance", "security", "design"])
        complexity = st.slider("Complexity (1-10)", 1, 10, 5)
        estimated_hours = st.number_input("Estimated Hours", min_value=0.5, value=8.0)
        days_to_deadline = st.number_input("Days until deadline", min_value=1, value=7)
        
        predict_btn = st.form_submit_button("Predict Priority")
        
        if predict_btn and task_desc:
            try:
                # Create sample task for prediction
                test_task = pd.DataFrame([{
                    'description': task_desc,
                    'category': task_category,
                    'complexity_score': complexity,
                    'estimated_hours': estimated_hours,
                    'days_until_deadline': days_to_deadline,
                    'status': 'pending'
                }])
                
                # Load model and predict
                model = TaskPriorityModel()
                model.load_model()
                
                # Preprocess test data
                preprocessor = TaskDataPreprocessor()
                processed_test = preprocessor.fit_transform(test_task)
                
                prediction = model.predict(processed_test)[0]
                
                st.success(f"Predicted Priority Score: {prediction:.2f}/10")
                
                # Priority interpretation
                if prediction >= 8:
                    priority_level = "🔴 Critical"
                elif prediction >= 6:
                    priority_level = "🟡 High"
                elif prediction >= 4:
                    priority_level = "🟢 Medium"
                else:
                    priority_level = "⚪ Low"
                
                st.write(f"Priority Level: {priority_level}")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")

def show_analytics_page(tasks_df, employee_profiles):
    """Show analytics and insights page"""
    
    st.header("Analytics & Insights")
    
    if tasks_df.empty:
        st.warning("No task data available for analytics.")
        return
    
    # Time-based analysis
    if 'created_at' in tasks_df.columns:
        st.subheader("Task Creation Trends")
        
        # Convert created_at to datetime
        tasks_df['created_at'] = pd.to_datetime(tasks_df['created_at'])
        tasks_df['date'] = tasks_df['created_at'].dt.date
        
        daily_tasks = tasks_df.groupby('date').size().reset_index(name='count')
        
        fig = px.line(
            daily_tasks,
            x='date',
            y='count',
            title="Daily Task Creation"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Task completion analysis
    if 'status' in tasks_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Task Status Distribution")
            status_counts = tasks_df['status'].value_counts()
            
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Task Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Category Performance")
            if 'category' in tasks_df.columns:
                category_status = pd.crosstab(tasks_df['category'], tasks_df['status'])
                
                fig = px.bar(
                    category_status.reset_index(),
                    x='category',
                    y=category_status.columns,
                    title="Task Status by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Team performance
    if employee_profiles:
        st.subheader("Team Performance")
        
        # Calculate team metrics
        total_capacity = sum(emp.get('max_capacity', 10) for emp in employee_profiles)
        total_workload = sum(emp.get('current_workload', 0) for emp in employee_profiles)
        avg_utilization = (total_workload / total_capacity * 100) if total_capacity > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Team Capacity", f"{total_capacity} hours")
        with col2:
            st.metric("Current Workload", f"{total_workload} hours")
        with col3:
            st.metric("Average Utilization", f"{avg_utilization:.1f}%")
        
        # Department analysis
        dept_data = {}
        for emp in employee_profiles:
            dept = emp.get('department', 'Unknown')
            if dept not in dept_data:
                dept_data[dept] = {'count': 0, 'total_capacity': 0, 'total_workload': 0}
            
            dept_data[dept]['count'] += 1
            dept_data[dept]['total_capacity'] += emp.get('max_capacity', 10)
            dept_data[dept]['total_workload'] += emp.get('current_workload', 0)
        
        dept_df = pd.DataFrame([
            {
                'Department': dept,
                'Employees': data['count'],
                'Capacity': data['total_capacity'],
                'Workload': data['total_workload'],
                'Utilization %': (data['total_workload'] / data['total_capacity'] * 100) if data['total_capacity'] > 0 else 0
            }
            for dept, data in dept_data.items()
        ])
        
        st.subheader("Department Overview")
        st.dataframe(dept_df, use_container_width=True)

if __name__ == "__main__":
    main() 