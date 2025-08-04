#!/usr/bin/env python3
"""
AI Task Management System - Streamlit Cloud Entry Point
Minimal version for reliable deployment
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

# Set page config - MUST be first
st.set_page_config(
    page_title="AI Task Management System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application - No database, no external dependencies"""
    
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
        # Main dashboard
        st.title("🤖 AI Task Management System")
        
        # Sidebar
        with st.sidebar:
            st.markdown("### 👤 User Info")
            st.write(f"**User:** {st.session_state.current_user}")
            st.write(f"**Role:** {st.session_state.user_role}")
            st.write(f"**Login Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            if st.button("🚪 Logout"):
                st.session_state.is_authenticated = False
                st.rerun()
        
        # Main content
        st.markdown("### 📊 Dashboard Overview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tasks", "25", "+5")
        
        with col2:
            st.metric("Completed", "18", "+3")
        
        with col3:
            st.metric("In Progress", "5", "-2")
        
        with col4:
            st.metric("Pending", "2", "+1")
        
        # Charts
        st.markdown("### 📈 Task Analytics")
        
        # Task Status Chart
        status_data = {
            'Status': ['Completed', 'In Progress', 'Pending', 'Overdue'],
            'Count': [18, 5, 2, 1],
            'Color': ['#00ff00', '#ffaa00', '#ff0000', '#ff0000']
        }
        df_status = pd.DataFrame(status_data)
        
        fig_status = px.pie(df_status, values='Count', names='Status', 
                           title="Task Status Distribution",
                           color_discrete_sequence=df_status['Color'])
        st.plotly_chart(fig_status, use_container_width=True)
        
        # Priority Chart
        priority_data = {
            'Priority': ['High', 'Medium', 'Low'],
            'Count': [8, 12, 5]
        }
        df_priority = pd.DataFrame(priority_data)
        
        fig_priority = px.bar(df_priority, x='Priority', y='Count',
                             title="Tasks by Priority",
                             color='Priority',
                             color_discrete_map={'High': '#ff0000', 'Medium': '#ffaa00', 'Low': '#00ff00'})
        st.plotly_chart(fig_priority, use_container_width=True)
        
        # Sample data table
        st.markdown("### 📋 Recent Tasks")
        sample_tasks = pd.DataFrame({
            'Task': ['Design UI Mockups', 'Database Optimization', 'API Integration', 'Testing', 'Documentation'],
            'Priority': ['High', 'Medium', 'High', 'Low', 'Medium'],
            'Status': ['In Progress', 'Completed', 'Pending', 'Completed', 'In Progress'],
            'Assignee': ['John Doe', 'Jane Smith', 'Mike Johnson', 'Sarah Wilson', 'Alex Brown'],
            'Due Date': ['2024-01-15', '2024-01-10', '2024-01-20', '2024-01-08', '2024-01-18']
        })
        st.dataframe(sample_tasks, use_container_width=True)
        
        # Quick actions
        st.markdown("### 🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Add New Task", key="add_task"):
                st.success("✅ Task creation feature coming soon!")
        
        with col2:
            if st.button("👥 Manage Employees", key="manage_employees"):
                st.success("✅ Employee management feature coming soon!")
        
        with col3:
            if st.button("📊 View Analytics", key="view_analytics"):
                st.success("✅ Advanced analytics feature coming soon!")
        
        # Additional info
        st.markdown("---")
        st.markdown("### ℹ️ System Information")
        st.info(f"""
        **Deployment Status:** ✅ Successfully deployed on Streamlit Cloud
        **Version:** 1.0.0
        **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        **Environment:** Streamlit Cloud
        **Database:** Demo Mode (No external database required)
        """)

# Run the application
if __name__ == "__main__":
    main() 