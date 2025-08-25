import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
from dreacon_ai_service import DreaconAIService

# --- CENTRALIZED LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="Dreacon AI Hub (Sentinel Engine)")

# Create a single, shared instance of our service to load models only once
service = DreaconAIService()

@app.post("/signup")
async def signup(
    username: str = Form(...),
    password: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Endpoint for new user registration.
    Requires username, password, and exactly 3 image files.
    """
    if len(files) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 image files are required for registration.")
    
    result = service.register_user(username, password, files)
    
    if result['status'] == 'error':
        raise HTTPException(status_code=409, detail=result['message']) # 409 Conflict
        
    return result

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """Endpoint for user login."""
    result = service.login_user(username, password)
    if result['status'] == 'error':
        raise HTTPException(status_code=401, detail=result['message']) # 401 Unauthorized
    return result

@app.post("/check_content")
async def check_content(file: UploadFile = File(...)):
    """
    The main security checkpoint endpoint for App2.
    Receives a media file and returns a final decision.
    """
    if not file.content_type.startswith('image/'):
        # In a full implementation, you would add video handling here
        raise HTTPException(status_code=400, detail="Currently, only image files are supported.")
        
    result = service.check_content(file)
    return result

@app.get("/")
def read_root():
    return {"message": "Welcome to the Dreacon AI Hub (Sentinel Engine) API"}

# To run this application:
# 1. Make sure you are in the project root directory ('sentinel_engine').
# 2. Make sure your virtual environment is active.
# 3. Run the command: uvicorn main_api:app --reload
