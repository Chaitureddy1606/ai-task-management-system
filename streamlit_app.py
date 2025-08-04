#!/usr/bin/env python3
"""
AI Task Management System - Streamlit Cloud Entry Point
Comprehensive version with full functionality
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import random

# Set page config - MUST be first
st.set_page_config(
    page_title="AI Task Management System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize demo data
def initialize_demo_data():
    """Initialize demo data for the application"""
    if 'demo_tasks' not in st.session_state:
        st.session_state.demo_tasks = [
            {
                'id': 1, 'title': 'Design UI Mockups', 'description': 'Create wireframes for new dashboard',
                'category': 'Design', 'priority': 'High', 'urgency_score': 8, 'complexity_score': 6,
                'business_impact': 7, 'estimated_hours': 16.0, 'days_until_deadline': 5,
                'status': 'In Progress', 'assigned_to': 'John Doe', 'created_at': '2024-01-10'
            },
            {
                'id': 2, 'title': 'Database Optimization', 'description': 'Optimize query performance',
                'category': 'Development', 'priority': 'Medium', 'urgency_score': 6, 'complexity_score': 8,
                'business_impact': 8, 'estimated_hours': 24.0, 'days_until_deadline': 8,
                'status': 'Completed', 'assigned_to': 'Jane Smith', 'created_at': '2024-01-08'
            },
            {
                'id': 3, 'title': 'API Integration', 'description': 'Integrate third-party payment API',
                'category': 'Development', 'priority': 'High', 'urgency_score': 9, 'complexity_score': 7,
                'business_impact': 9, 'estimated_hours': 20.0, 'days_until_deadline': 3,
                'status': 'Pending', 'assigned_to': 'Mike Johnson', 'created_at': '2024-01-12'
            },
            {
                'id': 4, 'title': 'Testing & QA', 'description': 'Comprehensive testing of new features',
                'category': 'Testing', 'priority': 'Medium', 'urgency_score': 5, 'complexity_score': 4,
                'business_impact': 6, 'estimated_hours': 12.0, 'days_until_deadline': 10,
                'status': 'Completed', 'assigned_to': 'Sarah Wilson', 'created_at': '2024-01-05'
            },
            {
                'id': 5, 'title': 'Documentation', 'description': 'Update user documentation',
                'category': 'Documentation', 'priority': 'Low', 'urgency_score': 3, 'complexity_score': 2,
                'business_impact': 4, 'estimated_hours': 8.0, 'days_until_deadline': 15,
                'status': 'In Progress', 'assigned_to': 'Alex Brown', 'created_at': '2024-01-15'
            }
        ]
    
    if 'demo_employees' not in st.session_state:
        st.session_state.demo_employees = [
            {
                'id': 'EMP001', 'name': 'John Doe', 'role': 'UI/UX Designer',
                'skills': 'Figma, Adobe XD, Prototyping', 'expertise_areas': 'UI Design, User Research',
                'current_workload': 70, 'max_capacity': 100, 'experience_years': 5,
                'location': 'New York', 'availability': 'Full-time'
            },
            {
                'id': 'EMP002', 'name': 'Jane Smith', 'role': 'Backend Developer',
                'skills': 'Python, SQL, Django', 'expertise_areas': 'Database Design, API Development',
                'current_workload': 85, 'max_capacity': 100, 'experience_years': 7,
                'location': 'San Francisco', 'availability': 'Full-time'
            },
            {
                'id': 'EMP003', 'name': 'Mike Johnson', 'role': 'Full Stack Developer',
                'skills': 'React, Node.js, MongoDB', 'expertise_areas': 'Frontend, Backend Integration',
                'current_workload': 60, 'max_capacity': 100, 'experience_years': 4,
                'location': 'Chicago', 'availability': 'Full-time'
            },
            {
                'id': 'EMP004', 'name': 'Sarah Wilson', 'role': 'QA Engineer',
                'skills': 'Selenium, JUnit, Manual Testing', 'expertise_areas': 'Test Automation, Quality Assurance',
                'current_workload': 45, 'max_capacity': 100, 'experience_years': 3,
                'location': 'Austin', 'availability': 'Full-time'
            },
            {
                'id': 'EMP005', 'name': 'Alex Brown', 'role': 'Technical Writer',
                'skills': 'Technical Writing, Markdown, API Docs', 'expertise_areas': 'Documentation, User Guides',
                'current_workload': 30, 'max_capacity': 100, 'experience_years': 2,
                'location': 'Remote', 'availability': 'Part-time'
            }
        ]

def show_dashboard():
    """Show the main dashboard"""
    st.markdown("### 📊 Dashboard Overview")
    
    # Metrics
    tasks_df = pd.DataFrame(st.session_state.demo_tasks)
    total_tasks = len(tasks_df)
    completed_tasks = len(tasks_df[tasks_df['status'] == 'Completed'])
    in_progress_tasks = len(tasks_df[tasks_df['status'] == 'In Progress'])
    pending_tasks = len(tasks_df[tasks_df['status'] == 'Pending'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tasks", total_tasks, "+5")
    
    with col2:
        st.metric("Completed", completed_tasks, "+3")
    
    with col3:
        st.metric("In Progress", in_progress_tasks, "-2")
    
    with col4:
        st.metric("Pending", pending_tasks, "+1")
    
    # Charts
    st.markdown("### 📈 Task Analytics")
    
    # Task Status Chart
    status_counts = tasks_df['status'].value_counts()
    fig_status = px.pie(values=status_counts.values, names=status_counts.index,
                       title="Task Status Distribution",
                       color_discrete_sequence=['#00ff00', '#ffaa00', '#ff0000', '#ff0000'])
    st.plotly_chart(fig_status, use_container_width=True)
    
    # Priority Chart
    priority_counts = tasks_df['priority'].value_counts()
    fig_priority = px.bar(x=priority_counts.index, y=priority_counts.values,
                          title="Tasks by Priority",
                          color=priority_counts.index,
                          color_discrete_map={'High': '#ff0000', 'Medium': '#ffaa00', 'Low': '#00ff00'})
    st.plotly_chart(fig_priority, use_container_width=True)
    
    # Task table
    st.markdown("### 📋 All Tasks")
    display_df = tasks_df[['title', 'category', 'priority', 'status', 'assigned_to', 'estimated_hours']].copy()
    st.dataframe(display_df, use_container_width=True)

def show_task_management():
    """Show task management page"""
    st.markdown("### 📝 Task Management")
    
    # Add new task
    with st.expander("➕ Add New Task", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            new_title = st.text_input("Task Title", key="new_task_title")
            new_description = st.text_area("Description", key="new_task_description")
            new_category = st.selectbox("Category", ["Development", "Design", "Testing", "Documentation", "Marketing"], key="new_task_category")
            new_priority = st.selectbox("Priority", ["High", "Medium", "Low"], key="new_task_priority")
        
        with col2:
            new_urgency = st.slider("Urgency Score", 1, 10, 5, key="new_task_urgency")
            new_complexity = st.slider("Complexity Score", 1, 10, 5, key="new_task_complexity")
            new_impact = st.slider("Business Impact", 1, 10, 5, key="new_task_impact")
            new_hours = st.number_input("Estimated Hours", min_value=1.0, max_value=100.0, value=8.0, key="new_task_hours")
        
        if st.button("Add Task", key="add_task_btn"):
            new_task = {
                'id': len(st.session_state.demo_tasks) + 1,
                'title': new_title,
                'description': new_description,
                'category': new_category,
                'priority': new_priority,
                'urgency_score': new_urgency,
                'complexity_score': new_complexity,
                'business_impact': new_impact,
                'estimated_hours': new_hours,
                'days_until_deadline': random.randint(1, 30),
                'status': 'Pending',
                'assigned_to': 'Unassigned',
                'created_at': datetime.now().strftime('%Y-%m-%d')
            }
            st.session_state.demo_tasks.append(new_task)
            st.success("✅ Task added successfully!")
            st.rerun()
    
    # Task list with actions
    st.markdown("### 📋 Manage Tasks")
    tasks_df = pd.DataFrame(st.session_state.demo_tasks)
    
    for idx, task in enumerate(st.session_state.demo_tasks):
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"**{task['title']}**")
                st.write(f"*{task['description']}*")
                st.write(f"Category: {task['category']} | Priority: {task['priority']}")
            
            with col2:
                new_status = st.selectbox("Status", ["Pending", "In Progress", "Completed"], 
                                        index=["Pending", "In Progress", "Completed"].index(task['status']),
                                        key=f"status_{task['id']}")
                if new_status != task['status']:
                    task['status'] = new_status
            
            with col3:
                st.write(f"**{task['assigned_to']}**")
            
            with col4:
                if st.button("Edit", key=f"edit_{task['id']}"):
                    st.session_state.editing_task = task['id']
            
            st.divider()

def show_employee_management():
    """Show employee management page"""
    st.markdown("### 👥 Employee Management")
    
    # Employee metrics
    employees_df = pd.DataFrame(st.session_state.demo_employees)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Employees", len(employees_df))
    with col2:
        avg_workload = employees_df['current_workload'].mean()
        st.metric("Avg Workload", f"{avg_workload:.1f}%")
    with col3:
        available_employees = len(employees_df[employees_df['current_workload'] < 80])
        st.metric("Available", available_employees)
    
    # Employee table
    st.markdown("### 👤 Employee List")
    display_emp_df = employees_df[['name', 'role', 'skills', 'current_workload', 'location']].copy()
    st.dataframe(display_emp_df, use_container_width=True)
    
    # Workload chart
    st.markdown("### 📊 Workload Distribution")
    fig_workload = px.bar(employees_df, x='name', y='current_workload',
                          title="Employee Workload",
                          color='current_workload',
                          color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig_workload, use_container_width=True)

def show_analytics():
    """Show analytics page"""
    st.markdown("### 📈 Advanced Analytics")
    
    tasks_df = pd.DataFrame(st.session_state.demo_tasks)
    employees_df = pd.DataFrame(st.session_state.demo_employees)
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Task Performance")
        
        # Completion rate
        completion_rate = (len(tasks_df[tasks_df['status'] == 'Completed']) / len(tasks_df)) * 100
        st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
        # Average complexity
        avg_complexity = tasks_df['complexity_score'].mean()
        st.metric("Avg Complexity", f"{avg_complexity:.1f}/10")
        
        # Average business impact
        avg_impact = tasks_df['business_impact'].mean()
        st.metric("Avg Business Impact", f"{avg_impact:.1f}/10")
    
    with col2:
        st.markdown("#### 👥 Team Performance")
        
        # Average workload
        avg_workload = employees_df['current_workload'].mean()
        st.metric("Avg Workload", f"{avg_workload:.1f}%")
        
        # Available capacity
        total_capacity = employees_df['max_capacity'].sum()
        used_capacity = employees_df['current_workload'].sum()
        available_capacity = total_capacity - used_capacity
        st.metric("Available Capacity", f"{available_capacity:.0f}%")
    
    # Advanced charts
    st.markdown("### 📊 Detailed Analytics")
    
    # Task complexity vs business impact
    fig_scatter = px.scatter(tasks_df, x='complexity_score', y='business_impact',
                             color='priority', size='estimated_hours',
                             title="Task Complexity vs Business Impact",
                             hover_data=['title'])
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Category analysis
    category_stats = tasks_df.groupby('category').agg({
        'estimated_hours': 'sum',
        'business_impact': 'mean',
        'complexity_score': 'mean'
    }).reset_index()
    
    fig_category = px.bar(category_stats, x='category', y='estimated_hours',
                          title="Hours by Category",
                          color='business_impact',
                          color_continuous_scale='Viridis')
    st.plotly_chart(fig_category, use_container_width=True)

def show_ai_insights():
    """Show AI-powered insights"""
    st.markdown("### 🤖 AI-Powered Insights")
    
    tasks_df = pd.DataFrame(st.session_state.demo_tasks)
    employees_df = pd.DataFrame(st.session_state.demo_employees)
    
    # AI Recommendations
    st.markdown("#### 💡 Smart Recommendations")
    
    # High priority tasks
    high_priority_tasks = tasks_df[tasks_df['priority'] == 'High']
    if not high_priority_tasks.empty:
        st.warning(f"⚠️ **{len(high_priority_tasks)} high-priority tasks** need immediate attention!")
        for _, task in high_priority_tasks.iterrows():
            st.write(f"• {task['title']} (Due in {task['days_until_deadline']} days)")
    
    # Overloaded employees
    overloaded_employees = employees_df[employees_df['current_workload'] > 80]
    if not overloaded_employees.empty:
        st.error(f"🚨 **{len(overloaded_employees)} employees** are overloaded!")
        for _, emp in overloaded_employees.iterrows():
            st.write(f"• {emp['name']} ({emp['current_workload']}% workload)")
    
    # Resource optimization
    st.markdown("#### 🔧 Resource Optimization")
    
    # Available employees for high-priority tasks
    available_employees = employees_df[employees_df['current_workload'] < 60]
    if not available_employees.empty and not high_priority_tasks.empty:
        st.success(f"✅ **{len(available_employees)} employees** are available for high-priority tasks")
        for _, emp in available_employees.iterrows():
            st.write(f"• {emp['name']} ({emp['role']}) - {emp['current_workload']}% workload")
    
    # Predictive analytics
    st.markdown("#### 🔮 Predictive Analytics")
    
    # Estimated completion time
    total_hours = tasks_df['estimated_hours'].sum()
    avg_workload = employees_df['current_workload'].mean()
    available_capacity = (100 - avg_workload) / 100
    
    if available_capacity > 0:
        estimated_days = total_hours / (8 * available_capacity * len(employees_df))
        st.info(f"📅 **Estimated completion time:** {estimated_days:.1f} days")
    
    # Risk assessment
    overdue_risk = len(tasks_df[tasks_df['days_until_deadline'] < 3])
    if overdue_risk > 0:
        st.error(f"⏰ **{overdue_risk} tasks** are at risk of being overdue!")

def main():
    """Main application"""
    
    # Initialize demo data
    initialize_demo_data()
    
    # Initialize session state
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
    
    if not st.session_state.is_authenticated:
        # Login page
        st.title("🤖 AI Task Management System")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔐 Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", key="login_btn"):
                if username == "admin" and password == "admin123":
                    st.session_state.is_authenticated = True
                    st.session_state.current_user = username
                    st.session_state.user_role = "admin"
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Use admin/admin123")
        
        with col2:
            st.subheader("📝 Quick Start")
            st.info("""
            **Demo Credentials:**
            - Username: `admin`
            - Password: `admin123`
            
            **Features:**
            - Task Management Dashboard
            - Employee Assignment
            - Analytics & Reports
            - AI-Powered Insights
            """)
    else:
        # Main application
        st.title("🤖 AI Task Management System")
        
        # Sidebar navigation
        with st.sidebar:
            st.markdown("### 👤 User Info")
            st.write(f"**User:** {st.session_state.current_user}")
            st.write(f"**Role:** {st.session_state.user_role}")
            st.write(f"**Login Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            st.markdown("---")
            st.markdown("### 🧭 Navigation")
            
            page = st.radio("Select Page", [
                "📊 Dashboard",
                "📝 Task Management", 
                "👥 Employee Management",
                "📈 Analytics",
                "🤖 AI Insights"
            ])
            
            st.markdown("---")
            if st.button("🚪 Logout"):
                st.session_state.is_authenticated = False
                st.rerun()
        
        # Page routing
        if page == "📊 Dashboard":
            show_dashboard()
        elif page == "📝 Task Management":
            show_task_management()
        elif page == "👥 Employee Management":
            show_employee_management()
        elif page == "📈 Analytics":
            show_analytics()
        elif page == "🤖 AI Insights":
            show_ai_insights()

# Run the application
if __name__ == "__main__":
    main() 