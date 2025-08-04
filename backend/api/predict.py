from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import requests
from typing import List, Any

router = APIRouter()

class PredictionInput(BaseModel):
    description: str

class SummarizeInput(BaseModel):
    text: str

class SkillMatchInput(BaseModel):
    description: str
    required_skills: List[str]
    employees: List[dict]

class NextActionsInput(BaseModel):
    description: str
    current_status: str = ""

class ChatInput(BaseModel):
    messages: List[dict]  # [{role: 'user'|'assistant', content: str}]

class BulkActionInput(BaseModel):
    command: str
    tasks: List[Any]

class AnalyzeNotesInput(BaseModel):
    notes: str

class NotificationsInput(BaseModel):
    tasks: List[Any]

class GenerateDescriptionInput(BaseModel):
    prompt: str

class AnalyticsInsightsInput(BaseModel):
    analytics: Any

@router.post('/predict-category')
def predict_category(input: PredictionInput):
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise HTTPException(status_code=500, detail='OpenRouter API key not set')
    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/predict',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'input': input.description
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f'OpenRouter API error: {response.text}')
        data = response.json()
        return {"category": data.get('category', 'Unknown')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error calling OpenRouter API: {str(e)}')

@router.post('/summarize')
def summarize(input: SummarizeInput):
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise HTTPException(status_code=500, detail='OpenRouter API key not set')
    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/summarize',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'text': input.text
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f'OpenRouter API error: {response.text}')
        data = response.json()
        return {"summary": data.get('summary', '')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error calling OpenRouter API: {str(e)}')

@router.post('/ai-suggest-employees')
def ai_suggest_employees(input: SkillMatchInput):
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise HTTPException(status_code=500, detail='OpenRouter API key not set')
    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/skill-match',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'description': input.description,
                'required_skills': input.required_skills,
                'employees': input.employees
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f'OpenRouter API error: {response.text}')
        data = response.json()
        return {"suggested_employees": data.get('suggested_employees', [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error calling OpenRouter API: {str(e)}')

@router.post('/ai-suggest-next-actions')
def ai_suggest_next_actions(input: NextActionsInput):
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise HTTPException(status_code=500, detail='OpenRouter API key not set')
    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/next-actions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'description': input.description,
                'current_status': input.current_status
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f'OpenRouter API error: {response.text}')
        data = response.json()
        return {"suggestions": data.get('suggestions', [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error calling OpenRouter API: {str(e)}')

@router.post('/chat')
def chat(input: ChatInput):
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise HTTPException(status_code=500, detail='OpenRouter API key not set')
    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/chat',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'messages': input.messages
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f'OpenRouter API error: {response.text}')
        data = response.json()
        return {"reply": data.get('reply', '')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error calling OpenRouter API: {str(e)}')

@router.post('/ai-bulk-action')
def ai_bulk_action(input: BulkActionInput):
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise HTTPException(status_code=500, detail='OpenRouter API key not set')
    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/bulk-action',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'command': input.command,
                'tasks': input.tasks
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f'OpenRouter API error: {response.text}')
        data = response.json()
        return {"action": data.get('action', {})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error calling OpenRouter API: {str(e)}')

@router.post('/analyze-notes')
def analyze_notes(input: AnalyzeNotesInput):
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise HTTPException(status_code=500, detail='OpenRouter API key not set')
    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/analyze-notes',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'notes': input.notes
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f'OpenRouter API error: {response.text}')
        data = response.json()
        return {"items": data.get('items', [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error calling OpenRouter API: {str(e)}')

@router.post('/ai-notifications')
def ai_notifications(input: NotificationsInput):
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise HTTPException(status_code=500, detail='OpenRouter API key not set')
    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/notifications',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'tasks': input.tasks
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f'OpenRouter API error: {response.text}')
        data = response.json()
        return {"notifications": data.get('notifications', [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error calling OpenRouter API: {str(e)}')

@router.post('/ai-generate-description')
def ai_generate_description(input: GenerateDescriptionInput):
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise HTTPException(status_code=500, detail='OpenRouter API key not set')
    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/generate-description',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'prompt': input.prompt
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f'OpenRouter API error: {response.text}')
        data = response.json()
        return {"description": data.get('description', '')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error calling OpenRouter API: {str(e)}')

@router.post('/ai-analytics-insights')
def ai_analytics_insights(input: AnalyticsInsightsInput):
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise HTTPException(status_code=500, detail='OpenRouter API key not set')
    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/analytics-insights',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'analytics': input.analytics
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f'OpenRouter API error: {response.text}')
        data = response.json()
        return {"insights": data.get('insights', '')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error calling OpenRouter API: {str(e)}') 