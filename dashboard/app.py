"""
Streamlit Dashboard for AI Task Management System
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import numpy as np

# Load models
classifier = joblib.load("models/classifier.pkl")
priority_model = joblib.load("models/priority_model.pkl")

# Load employee profile
with open("data/employee_profiles.json", "r") as f:
    import json
    employee_profiles = json.load(f)

# Import necessary tools for React components
from dash_local_react_components import load_react_component

# Load custom React components
ReactComponentHero = load_react_component(app, 'components', 'HeroSection.js')
ReactComponentTabs = load_react_component(app, 'components', 'TabsSection.js')

# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "AI Task Manager"

# Custom CSS for better styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>AI Task Manager</title>
        <style>
            .dash-tab {
                background-color: #f8f9fa !important;
                border: 1px solid #dee2e6 !important;
                border-radius: 8px 8px 0 0 !important;
                margin-right: 5px !important;
                padding: 12px 20px !important;
                font-weight: 500 !important;
                color: #495057 !important;
            }
            .dash-tab--selected {
                background-color: #007bff !important;
                color: white !important;
                border-color: #007bff !important;
            }
            .dash-tab:hover {
                background-color: #e9ecef !important;
                color: #495057 !important;
            }
            .dash-tab--selected:hover {
                background-color: #0056b3 !important;
                color: white !important;
            }
            .upload-area {
                border: 2px dashed #007bff !important;
                border-radius: 10px !important;
                background-color: #f8f9fa !important;
                transition: all 0.3s ease !important;
            }
            .upload-area:hover {
                border-color: #0056b3 !important;
                background-color: #e3f2fd !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# In the app.layout, within ReactComponentTabs, replace the analytics tab content with AnalyticsSection
app.layout = html.Div([
    ReactComponentHero(),
    ReactComponentTabs([
        dcc.Tab(label='📤 Upload & Predict', children=[
            html.H3("Upload Tasks (.csv)"),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.I(className="fas fa-cloud-upload-alt", style={'fontSize': '2rem', 'color': '#007bff', 'marginBottom': '10px'}),
                    html.Br(),
                    html.Span("Drag and Drop or ", style={'fontSize': '1.1rem'}),
                    html.A("Select Files", style={'color': '#007bff', 'textDecoration': 'underline'}),
                    html.Br(),
                    html.Span("(CSV format)", style={'fontSize': '0.9rem', 'color': '#6c757d', 'marginTop': '5px'})
                ]),
                style={
                    'width': '100%',
                    'height': '120px',
                    'lineHeight': '60px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'margin': '20px 0',
                    'backgroundColor': '#f8f9fa',
                    'borderColor': '#007bff',
                    'transition': 'all 0.3s ease'
                },
                multiple=False
            ),
            html.Div(id='file-name-display'),
            html.Div(id='status-msg'),
            
            # Assignment Method Toggle
            html.Div([
                html.Label("Assignment Method:", style={'fontWeight': 'bold', 'marginTop': '20px'}),
                dcc.RadioItems(
                    id='assign-method',
                    options=[
                        {'label': ' 🎯 Rule-based', 'value': 'rule'},
                        {'label': ' 🤖 ML-based (coming soon)', 'value': 'ml'}
                    ],
                    value='rule',
                    inline=True,
                    style={'marginTop': '10px'}
                )
            ], style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px'}),
            
            # Category Filter Dropdown
            html.Div([
                html.Label("Filter by Category:", style={'fontWeight': 'bold', 'marginTop': '20px'}),
                dcc.Dropdown(
                    id='category-filter',
                    placeholder='Select task categories',
                    multi=True,
                    style={'marginTop': '10px'}
                )
            ], style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px'}),
            
            html.Div(id='output-data-upload')
        ]),

        dcc.Tab(label='📊 Analytics', children=[
            AnalyticsSection(
                metrics=[
                    {'label': 'Total Tasks', 'value': 100},  # Demo value; replace with dynamic data later
                    {'label': 'High Priority', 'value': 30},  # Demo value; replace with dynamic data later
                    {'label': 'Team Utilization', 'value': '75%'},  # Demo value
                    {'label': 'Avg Completion Time', 'value': '5.2 days'}  # Demo value
                ],
                children=[
                    html.Div([
                        html.H3("Comprehensive Analytics Dashboard"),
                        html.Div([
                            html.Div([
                                html.H4("Total Tasks"),
                                html.H2(id="total-tasks")
                            ], style={'display': 'inline-block', 'width': '24%', 'margin': '1%'}),
                            html.Div([
                                html.H4("High Priority"),
                                html.H2(id="high-priority")
                            ], style={'display': 'inline-block', 'width': '24%', 'margin': '1%'}),
                            html.Div([
                                html.H4("Team Utilization"),
                                html.H2(id="team-utilization")
                            ], style={'display': 'inline-block', 'width': '24%', 'margin': '1%'}),
                            html.Div([
                                html.H4("Avg Completion Time"),
                                html.H2(id="avg-completion")
                            ], style={'display': 'inline-block', 'width': '24%', 'margin': '1%'})
                        ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between'}),
                    ], className="mb-4"),
                    html.Div([
                        html.Div([
                            dcc.Graph(id='priority-distribution')
                        ], style={'width': '48%', 'display': 'inline-block'}),
                        html.Div([
                            dcc.Graph(id='category-distribution')
                        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                    ]),
                    html.Div([
                        html.Div([
                            dcc.Graph(id='workload-chart')
                        ], style={'width': '48%', 'display': 'inline-block'}),
                        html.Div([
                            dcc.Graph(id='performance-trend')
                        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                    ]),
                    html.Div([
                        html.Div([
                            dcc.Graph(id='employee-skills-radar')
                        ], style={'width': '48%', 'display': 'inline-block'}),
                        html.Div([
                            dcc.Graph(id='task-completion-timeline')
                        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                    ]),
                    html.Div([
                        html.H4("Task Status Timeline"),
                        dcc.Graph(id='task-timeline')
                    ])
                ]
            )
        ])
    ]),
    html.Hr(style={'margin': '20px 0', 'borderColor': '#e9ecef'})
])

# --- Analytics Data Preparation (Simulated for Demo) ---
def get_demo_task_data():
    # Simulate a DataFrame of tasks
    np.random.seed(42)
    n_tasks = 100
    employees = [emp['name'] for emp in employee_profiles]
    categories = ['Bug', 'Feature', 'Improvement', 'Research']
    priorities = ['Low', 'Medium', 'High']
    statuses = ['Pending', 'In Progress', 'Completed']

    df = pd.DataFrame({
        'task_id': range(1, n_tasks + 1),
        'employee': np.random.choice(employees, n_tasks),
        'category': np.random.choice(categories, n_tasks),
        'priority': np.random.choice(priorities, n_tasks, p=[0.2, 0.5, 0.3]),
        'status': np.random.choice(statuses, n_tasks, p=[0.2, 0.5, 0.3]),
        'completion_time': np.random.normal(5, 2, n_tasks).clip(1, 10),
        'created_at': pd.date_range('2024-01-01', periods=n_tasks, freq='D')
    })
    return df

# --- Analytics Callbacks ---
@app.callback(
    Output('total-tasks', 'children'),
    Output('high-priority', 'children'),
    Output('team-utilization', 'children'),
    Output('avg-completion', 'children'),
    Output('priority-distribution', 'figure'),
    Output('category-distribution', 'figure'),
    Output('workload-chart', 'figure'),
    Output('performance-trend', 'figure'),
    Output('employee-skills-radar', 'figure'),
    Output('task-completion-timeline', 'figure'),
    Output('task-timeline', 'figure'),
    Input('analytics-content', 'id')
)
def update_analytics(_):
    df = get_demo_task_data()
    # Metrics
    total_tasks = len(df)
    high_priority = (df['priority'] == 'High').sum()
    team_util = f"{(df['employee'].nunique() / len(employee_profiles)) * 100:.1f}%"
    avg_completion = f"{df['completion_time'].mean():.2f} days"

    # Priority Distribution
    fig_priority = px.pie(df, names='priority', title='Task Priority Distribution')

    # Category Distribution
    fig_category = px.bar(df['category'].value_counts().reset_index(),
                          x='index', y='category',
                          labels={'index': 'Category', 'category': 'Count'},
                          title='Task Category Distribution')

    # Workload Chart
    workload = df.groupby('employee').size().reset_index(name='tasks')
    fig_workload = px.bar(workload, x='employee', y='tasks', title='Workload per Employee')

    # Performance Trend
    perf_trend = df.groupby('created_at')['completion_time'].mean().reset_index()
    fig_perf = px.line(perf_trend, x='created_at', y='completion_time', title='Avg Completion Time Over Time')

    # Employee Skills Radar (simulated)
    skill_set = set(skill for emp in employee_profiles for skill in emp.get('skills', []))
    radar_data = []
    for emp in employee_profiles:
        radar_data.append({
            'employee': emp['name'],
            **{skill: int(skill in emp.get('skills', [])) for skill in skill_set}
        })
    radar_df = pd.DataFrame(radar_data)
    fig_radar = go.Figure()
    for i, row in radar_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=row[list(skill_set)].values,
            theta=list(skill_set),
            fill='toself',
            name=row['employee']
        ))
    fig_radar.update_layout(title='Employee Skills Radar', polar=dict(radialaxis=dict(visible=True)))

    # Task Completion Timeline
    completed = df[df['status'] == 'Completed']
    timeline = completed.groupby('created_at').size().reset_index(name='completed')
    fig_timeline = px.bar(timeline, x='created_at', y='completed', title='Tasks Completed Over Time')

    # Task Status Timeline
    status_timeline = df.groupby(['created_at', 'status']).size().reset_index(name='count')
    fig_status_timeline = px.area(status_timeline, x='created_at', y='count', color='status', title='Task Status Timeline')

    return (str(total_tasks), str(high_priority), team_util, avg_completion,
            fig_priority, fig_category, fig_workload, fig_perf, fig_radar, fig_timeline, fig_status_timeline)

# --- Existing Upload & Predict logic remains unchanged ---

def preprocess_and_predict(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")  # Assume saved during training

    # Preprocess
    X = tfidf.transform(df['description'])
    df['category'] = classifier.predict(X)
    df['priority'] = priority_model.predict(X)
    return df

def assign_task(task_row):
    best_employee = None
    max_match = -1
    for emp in employee_profiles:
        if emp['current_tasks'] >= emp['max_tasks']:
            continue
        match = len(set(emp['skills']) & set(task_row['description'].lower().split()))
        if match > max_match:
            best_employee = emp['name']
            max_match = match
    return best_employee if best_employee else "Unassigned"

@app.callback(
    Output('output-data-upload', 'children'),
    Output('file-name-display', 'children'),
    Output('status-msg', 'children'),
    Output('category-filter', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('assign-method', 'value'),
    State('category-filter', 'value')
)
def update_output(contents, filename, assign_method, category_filter):
    if contents is None:
        return html.Div(), html.Div(), html.Div(), []

    import base64
    import io

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # File name display
    file_display = html.Div(f"✅ Uploaded: {filename}", style={
        'color': '#28a745',
        'fontWeight': 'bold',
        'margin': '10px 0',
        'padding': '10px',
        'backgroundColor': '#d4edda',
        'borderRadius': '5px',
        'border': '1px solid #c3e6cb'
    })

    # Status message
    status_msg = html.Div("✅ File uploaded successfully! Processing tasks...", style={
        'color': '#155724',
        'backgroundColor': '#d4edda',
        'border': '1px solid #c3e6cb',
        'borderRadius': '5px',
        'padding': '10px',
        'margin': '10px 0'
    })

    # Process the data
    df = preprocess_and_predict(df)
    
    # Apply assignment method
    if assign_method == 'rule':
        df['assigned_to'] = df.apply(assign_task, axis=1)
    else:
        df['assigned_to'] = "ML Assignment (Coming Soon)"

    # Apply category filter if selected
    if category_filter:
        df = df[df['category'].isin(category_filter)]

    # Create category options for dropdown
    category_options = [{'label': cat, 'value': cat} for cat in df['category'].unique()]

    # Create visualizations
    fig_priority = px.pie(df, names='priority', title='Task Priority Distribution')
    fig_category = px.bar(df['category'].value_counts().reset_index(),
                          x='index', y='category',
                          title='Task Category Distribution')

    return html.Div([
        html.H4("Predicted & Assigned Tasks"),
        html.Div([
            html.P(f"📊 Total Tasks: {len(df)}", style={'fontWeight': 'bold'}),
            html.P(f"🎯 Assignment Method: {assign_method.upper()}", style={'fontWeight': 'bold'}),
            html.P(f"📁 File: {filename}", style={'fontWeight': 'bold'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '20px'}),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={
                'backgroundColor': '#007bff',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f8f9fa'
                }
            ]
        ),
        html.Br(),
        html.Div([
            html.Div([
                dcc.Graph(figure=fig_priority)
            ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(figure=fig_category)
            ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
        ])
    ]), file_display, status_msg, category_options

if __name__ == '__main__':
    app.run(debug=True) 