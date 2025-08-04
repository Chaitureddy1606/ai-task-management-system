from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.employee import Base, Employee

# Provided employee data
employees = [
    {"employee_id": "E001", "name": "Samantha Russell", "skills": ["database", "backend", "testing", "cloud", "UX"], "current_tasks": 4},
    {"employee_id": "E002", "name": "Sandra Casey", "skills": ["security", "UX", "data analysis", "cloud"], "current_tasks": 3},
    {"employee_id": "E003", "name": "Raymond Mercado", "skills": ["testing", "database", "DevOps", "data analysis"], "current_tasks": 1},
    {"employee_id": "E004", "name": "Cynthia Ho", "skills": ["NLP", "backend", "DevOps", "API", "cloud"], "current_tasks": 1},
    {"employee_id": "E005", "name": "Diana Jackson", "skills": ["data analysis", "ML", "database", "DevOps", "UX"], "current_tasks": 1},
    {"employee_id": "E006", "name": "Amy English", "skills": ["NLP", "backend", "cloud", "UI"], "current_tasks": 3},
    {"employee_id": "E007", "name": "James Pennington", "skills": ["security", "testing", "API", "DevOps", "data analysis"], "current_tasks": 2},
    {"employee_id": "E008", "name": "Alexis Woods", "skills": ["security", "database", "testing"], "current_tasks": 2},
    {"employee_id": "E009", "name": "Jacqueline Franco", "skills": ["UX", "security", "DevOps"], "current_tasks": 4},
    {"employee_id": "E010", "name": "Leslie Ray", "skills": ["UX", "security", "testing"], "current_tasks": 3},
    {"employee_id": "E011", "name": "Kathleen Steele", "skills": ["bugfix", "UX", "testing", "frontend", "API"], "current_tasks": 3},
    {"employee_id": "E012", "name": "Michael Jones", "skills": ["security", "data analysis", "API"], "current_tasks": 5},
    {"employee_id": "E013", "name": "Catherine Ellis", "skills": ["DevOps", "database", "data analysis", "testing", "bugfix", "ML"], "current_tasks": 2},
    {"employee_id": "E014", "name": "Dawn Bell", "skills": ["security", "testing", "ML", "frontend"], "current_tasks": 0},
    {"employee_id": "E015", "name": "Sandra Clark", "skills": ["API", "security", "database", "UI", "testing"], "current_tasks": 3},
    {"employee_id": "E016", "name": "Sean Garza MD", "skills": ["database", "API", "testing", "data analysis", "security", "frontend"], "current_tasks": 5},
    {"employee_id": "E017", "name": "Harry Collins", "skills": ["API", "NLP", "ML", "frontend", "testing"], "current_tasks": 5},
    {"employee_id": "E018", "name": "Nicole Smith", "skills": ["frontend", "testing", "UX", "UI"], "current_tasks": 3},
    {"employee_id": "E019", "name": "Amy Wiggins", "skills": ["bugfix", "data analysis", "NLP", "backend"], "current_tasks": 0},
    {"employee_id": "E020", "name": "Jacob Hayden", "skills": ["security", "NLP", "cloud", "ML", "UI", "testing"], "current_tasks": 2},
    {"employee_id": "E021", "name": "Jill Martinez", "skills": ["NLP", "DevOps", "security"], "current_tasks": 3},
    {"employee_id": "E022", "name": "Mark Becker", "skills": ["frontend", "bugfix", "UI"], "current_tasks": 5},
    {"employee_id": "E023", "name": "Shawn Smith", "skills": ["NLP", "data analysis", "cloud"], "current_tasks": 4},
    {"employee_id": "E024", "name": "Michael Blackburn", "skills": ["DevOps", "database", "API", "security", "bugfix"], "current_tasks": 1},
    {"employee_id": "E025", "name": "Noah Robinson", "skills": ["database", "frontend", "ML", "security", "UI"], "current_tasks": 2}
]

engine = create_engine('sqlite:///employees.db')
Session = sessionmaker(bind=engine)
session = Session()

Base.metadata.create_all(engine)

for emp in employees:
    if not session.query(Employee).filter_by(employee_id=emp['employee_id']).first():
        session.add(Employee(**emp))
session.commit()
session.close()
print('Employee database created and populated.') 