# AI Task Management System 🤖

A comprehensive AI-powered task management system with intelligent employee assignment, real-time analytics, and advanced features.

## 🚀 Live Demo

**Streamlit Cloud Deployment:** [Your Streamlit App URL]
**GitHub Repository:** https://github.com/Chaitureddy1606/ai-task-management-system

## ✨ Features

### 🎯 Core Features
- **Auto Task Assignment** - Upload any CSV format and auto-assign to employees
- **User Authentication** - Sign up, login, role management
- **Real-time Notifications** - Live updates and pending task alerts
- **Mobile Responsive Design** - Works on all devices
- **Task Status Management** - Track progress, urgency, complexity
- **Comments System** - Team collaboration on tasks
- **Dependencies Management** - Task prerequisites and relationships

### 🤖 Advanced AI Features
- **Google Gemini AI Integration** - Smart task analysis and recommendations
- **Natural Language Queries** - Ask questions in plain English
- **Predictive Analytics** - AI-driven insights and forecasting
- **Smart Prioritization** - AI-powered task prioritization

### 📊 Analytics & Reporting
- **Advanced Analytics** - KPIs, productivity metrics, custom reports
- **Team Chat System** - Built-in communication
- **Gantt Charts** - Visual project timelines
- **Export Functionality** - Download reports and data

### 👥 Employee Management
- **CSV/JSON Upload** - Bulk employee import with auto-column detection
- **AI Employee Assignment** - Smart matching based on skills and workload
- **Employee Analytics** - Performance tracking and insights

## 🛠️ Installation

### Local Development

1. **Clone the repository:**
```bash
git clone https://github.com/Chaitureddy1606/ai-task-management-system.git
cd ai-task-management-system
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up configuration:**
```bash
# Copy config template
cp config_template.py config.py
# Edit config.py with your API keys
```

4. **Run the application:**
```bash
streamlit run streamlit_dashboard.py --server.port 8518
```

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account
2. **Connect to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your forked repository
   - Set the path to: `streamlit_app.py`
   - Click "Deploy"

3. **Configure environment variables** in Streamlit Cloud:
   - Go to your app settings
   - Add your Gemini API key as an environment variable

## 📁 Project Structure

```
ai-task-management-system/
├── streamlit_dashboard.py      # Main application
├── streamlit_app.py           # Streamlit Cloud entry point
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── .streamlit/config.toml    # Streamlit configuration
├── deploy_automation.sh      # Auto deployment script
├── sample_employees.csv      # Sample employee data
├── sample_employees.json     # Sample employee data
└── README.md                # This file
```

## 🔧 Configuration

### Environment Variables

Create a `config.py` file with your settings:

```python
# Gemini API Configuration
GEMINI_API_KEY = "your_gemini_api_key_here"

# App Configuration
APP_TITLE = "AI Task Management System"
APP_ICON = "🤖"

# Feature Flags
ENABLE_GEMINI = True
ENABLE_ADVANCED_AI = True
ENABLE_TEAM_CHAT = True
ENABLE_GANTT_CHARTS = True
```

### Streamlit Cloud Environment Variables

Add these in your Streamlit Cloud app settings:
- `GEMINI_API_KEY`: Your Google Gemini API key

## 🚀 Auto Deployment

### Automatic Git & Streamlit Updates

The project includes an automation script that automatically commits changes and triggers Streamlit Cloud deployment:

```bash
# Run the auto deployment script
./deploy_automation.sh
```

This script will:
1. ✅ Check for changes in your local files
2. ✅ Add all changes to Git
3. ✅ Create a commit with timestamp
4. ✅ Push to your GitHub repository
5. ✅ Trigger automatic Streamlit Cloud deployment

### Manual Deployment

If you prefer manual deployment:

```bash
# Commit and push changes
git add .
git commit -m "Update: [describe your changes]"
git push origin main

# Streamlit Cloud will automatically deploy from your GitHub repository
```

## 📊 Usage

### 1. **Employee Management**
- Navigate to "Employee Management" in the sidebar
- Use "Upload Employees" tab to import CSV/JSON files
- System automatically detects and maps columns
- Save employees to database for AI assignment

### 2. **Task Upload & Assignment**
- Go to "Upload Tasks" page
- Upload CSV files with task data
- System automatically assigns tasks to employees
- View assignments in "Task Analysis"

### 3. **AI Features**
- Use "Advanced AI" tab for predictive analytics
- Try natural language queries in "Natural Language" tab
- Explore AI insights and recommendations

### 4. **Analytics & Reporting**
- View comprehensive analytics in "Advanced Analytics"
- Export reports and data
- Monitor team performance and productivity

## 🔑 API Keys Setup

### Google Gemini API

1. **Get API Key:**
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key

2. **Configure:**
   - Add to `config.py` for local development
   - Add as environment variable in Streamlit Cloud

## 📝 File Formats

### Employee Upload (CSV/JSON)
Supported columns:
- **Name:** name, employee_name, full_name, first_name, last_name
- **Role:** role, position, job_title, title, designation
- **Department:** department, dept, team, division, unit
- **Location:** location, city, office, site, branch
- **Email:** email, email_address, e-mail, contact_email
- **Phone:** phone, phone_number, mobile, contact, telephone
- **Experience:** experience_years, experience, years_experience
- **Skills:** skills, skill_set, competencies, expertise
- **Workload:** current_workload, workload, current_load
- **Capacity:** max_capacity, capacity, max_workload
- **Salary:** salary, compensation, pay, wage, income
- **Hire Date:** hire_date, start_date, joining_date
- **Manager:** manager, supervisor, reporting_to, boss

### Task Upload (CSV)
Supported columns:
- **Title:** title, task_title, name, task_name
- **Description:** description, desc, details, task_description
- **Category:** category, type, task_category, task_type
- **Priority:** priority, task_priority, importance
- **Urgency:** urgency, urgency_score, criticality
- **Complexity:** complexity, complexity_score, difficulty
- **Business Impact:** business_impact, impact, value
- **Estimated Hours:** estimated_hours, hours, time_estimate

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues:

1. **Check the logs** in Streamlit Cloud
2. **Verify configuration** in `config.py`
3. **Test locally** before deploying
4. **Create an issue** on GitHub

## 🎯 Roadmap

- [ ] Advanced machine learning models
- [ ] Integration with external project management tools
- [ ] Mobile app development
- [ ] Advanced reporting features
- [ ] Multi-language support

---

**Made with ❤️ using Streamlit and AI** 