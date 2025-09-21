# 1. Import necessary libraries
import os
from datetime import datetime, timedelta, timezone
import jwt
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field, EmailStr
from passlib.context import CryptContext
from openai import OpenAI
from fastapi.responses import JSONResponse

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore

# --- Application Setup ---
app = FastAPI()
load_dotenv()

# --- CORS Middleware ---
origins = ["null", "*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Firebase Admin SDK Initialization ---
try:
    if not firebase_admin._apps:
        cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_PATH")
        if not cred_path or not os.path.exists(cred_path):
            raise ValueError(f"Service account key file not found at path: {cred_path}.")
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialized successfully.")
except Exception as e:
    print(f"FATAL ERROR: Firebase Admin SDK initialization failed: {e}")
    exit()
db = firestore.client()

# --- Security and API Clients ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Pydantic Models ---
class UserAuth(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)

class ProfileUpdate(BaseModel):
    displayName: str = Field(..., min_length=3, max_length=30)
    avatar: str

class UserDetailsUpdate(BaseModel):
    displayName: str = Field(..., min_length=3, max_length=50)
    phoneNumber: str | None = Field(None, max_length=20)

class QuizData(BaseModel):
    skills: str
    interests: str
    experience: str
    
class CompleteStepData(BaseModel):
    stepIndex: int

class StepDetailRequest(BaseModel):
    stepTitle: str
    stepDescription: str

class JobSearchRequest(BaseModel):
    job_title: str

class DetailRequest(BaseModel):
    title: str
    context: str
    
# --- Helper Functions ---
def verify_password(plain_password, hashed_password): return pwd_context.verify(plain_password, hashed_password)
def get_password_hash(password): return pwd_context.hash(password)
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user_email(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None: raise credentials_exception
        return email
    except jwt.PyJWTError:
        raise credentials_exception

# --- API Endpoints ---
@app.post("/register")
async def register(user: UserAuth):
    user_doc = db.collection('users').document(user.email).get()
    if user_doc.exists:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    db.collection('users').document(user.email).set({
        "email": user.email,
        "hashed_password": hashed_password,
        "profile_setup_completed": False,
        "quiz_completed": False,
        "xp": 0,
        "level": 1,
        "completed_steps": [],
        "avatar": "avatar-1",
        "displayName": user.email.split('@')[0],
    })
    return {"message": "User registered successfully"}

@app.post("/token")
async def login(user: UserAuth):
    user_doc = db.collection('users').document(user.email).get()
    if not user_doc.exists or not verify_password(user.password, user_doc.to_dict()['hashed_password']):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token = create_access_token({"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user_email: str = Depends(get_current_user_email)):
    user_doc = db.collection('users').document(current_user_email).get()
    if user_doc.exists: return user_doc.to_dict()
    raise HTTPException(status_code=404, detail="User not found")

@app.post("/update-profile")
async def update_profile(profile_data: ProfileUpdate, current_user_email: str = Depends(get_current_user_email)):
    user_ref = db.collection('users').document(current_user_email)
    user_ref.update({"displayName": profile_data.displayName, "avatar": profile_data.avatar, "profile_setup_completed": True})
    return {"message": "Profile updated successfully"}
    
@app.post("/update-user-details")
async def update_user_details(details: UserDetailsUpdate, current_user_email: str = Depends(get_current_user_email)):
    user_ref = db.collection('users').document(current_user_email)
    user_ref.update({
        "displayName": details.displayName,
        "phoneNumber": details.phoneNumber
    })
    return {"message": "Details updated successfully"}

@app.post("/submit-quiz")
async def submit_quiz(quiz_data: QuizData, current_user_email: str = Depends(get_current_user_email)):
    prompt = f"""
    Act as an expert career navigator AI. A student has provided their profile:
    - Skills: {quiz_data.skills}
    - Interests: {quiz_data.interests}
    - Experience: {quiz_data.experience}

    Your output MUST be a single, minified JSON object.
    The JSON object must have these keys: "careerTitle", "summary", "roadmap", "skillGaps", and "recommendations".

    - "careerTitle": A string for the recommended career.
    - "summary": A 2-3 sentence summary for this career path, in Markdown format.
    - "roadmap": An array of objects for the learning steps. Each object must have "title", "description", "type" ("Course", "Project", etc.), and "xp".
    - "skillGaps": An array of 3-4 strings listing the most important skills the user needs to learn.
    - "recommendations": An array of exactly 3 objects. Each object must have "title", "type" ("Course" or "Internship"), and a real "url". Generate 2 courses and 1 internship.
    """
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "You provide career roadmaps in a specific JSON format."}, {"role": "user", "content": prompt}])
        ai_response_json = json.loads(response.choices[0].message.content)

        ai_roadmap_data = {
            "careerTitle": ai_response_json.get("careerTitle"),
            "summary": ai_response_json.get("summary"),
            "roadmap": ai_response_json.get("roadmap")
        }

        user_ref = db.collection('users').document(current_user_email)
        user_ref.update({
            "quiz_data": quiz_data.model_dump(), 
            "ai_roadmap": ai_roadmap_data,
            "skill_gaps": ai_response_json.get("skillGaps", []),
            "recommendations": ai_response_json.get("recommendations", []),
            "quiz_completed": True,
            "last_updated": datetime.now(timezone.utc), 
            "xp": 0, 
            "level": 1, 
            "completed_steps": []
        })

        return {"message": "New roadmap generated!", "roadmap": ai_roadmap_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/get-step-details")
async def get_step_details(request: StepDetailRequest, current_user_email: str = Depends(get_current_user_email)):
    prompt = f"""
    You are an expert career counselor. For the step "{request.stepTitle}", provide a detailed guide in Markdown including an overview, key topics, actionable steps, and recommend 2-3 real, high-quality online courses with generic search URLs.
    """
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "You are a helpful career counselor AI."}, {"role": "user", "content": prompt}])
        return {"details": response.choices[0].message.content} 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating step details: {str(e)}")

@app.post("/complete-step")
async def complete_step(step_data: CompleteStepData, current_user_email: str = Depends(get_current_user_email)):
    user_ref = db.collection('users').document(current_user_email)
    user_doc = user_ref.get()
    if not user_doc.exists: raise HTTPException(status_code=404, detail="User not found")
    user_data = user_doc.to_dict()
    roadmap = user_data.get("ai_roadmap", {}).get("roadmap", [])
    completed_steps = user_data.get("completed_steps", [])
    step_index = step_data.stepIndex
    if not (0 <= step_index < len(roadmap)): raise HTTPException(status_code=400, detail="Invalid step index")
    if step_index in completed_steps: return {"message": "Step already completed"}
    xp_gained = roadmap[step_index].get("xp", 50)
    new_xp = user_data.get("xp", 0) + xp_gained
    completed_steps.append(step_index)
    new_level = int(new_xp / 250) + 1
    user_ref.update({"xp": new_xp, "level": new_level, "completed_steps": completed_steps})
    return {"message": "Step completed!", "new_xp": new_xp, "new_level": new_level}

@app.post("/jobs/search")
async def search_jobs(request: JobSearchRequest, current_user_email: str = Depends(get_current_user_email)):
    prompt = f"""
    You are a job search assistant. Based on your training data, provide 3 example job postings for the role of "{request.job_title}".
    For each job, provide a realistic job_title, company_name, and a url.
    Return the result as a minified JSON array of objects.
    """
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", response_format={"type": "json_object"}, messages=[{"role": "system", "content": "You provide job posting examples as a raw JSON array."}, {"role": "user", "content": prompt}])
        # The AI is asked to return an array, so we parse its content which is a JSON string.
        # The response from OpenAI might be wrapped in a dictionary, e.g., {"jobs": [...]}, so we adapt.
        response_data = json.loads(response.choices[0].message.content)
        if isinstance(response_data, dict) and 'jobs' in response_data:
            return response_data['jobs']
        return response_data # Assuming it returns a raw array as requested
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching for jobs: {str(e)}")

@app.get("/api/skill-gaps")
async def get_skill_gaps(current_user_email: str = Depends(get_current_user_email)):
    user_doc = db.collection('users').document(current_user_email).get()
    if not user_doc.exists or not user_doc.to_dict().get('quiz_completed'):
        raise HTTPException(status_code=404, detail="User has not completed the quiz yet.")
    user_data = user_doc.to_dict()
    return {"skill_gaps": user_data.get("skill_gaps", [])}

@app.get("/api/recommendations")
async def get_recommendations(current_user_email: str = Depends(get_current_user_email)):
    user_doc = db.collection('users').document(current_user_email).get()
    if not user_doc.exists or not user_doc.to_dict().get('quiz_completed'):
        raise HTTPException(status_code=404, detail="User has not completed the quiz yet.")
    user_data = user_doc.to_dict()
    return {"recommendations": user_data.get("recommendations", [])}

@app.post("/get-details")
async def get_details(request: DetailRequest, current_user_email: str = Depends(get_current_user_email)):
    user_doc = db.collection('users').document(current_user_email).get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_data = user_doc.to_dict()
    career_title = user_data.get("ai_roadmap", {}).get("careerTitle", "their chosen career path")

    prompt = f"""
    Act as an expert career counselor. A user is asking for more information about "{request.title}" in the context of "{request.context}" for their goal of becoming a {career_title}.
    Provide a detailed but easy-to-understand explanation in Markdown format.
    - If it's a skill, explain what it is and why it's crucial for their career.
    - If it's a course or internship, briefly describe what the user would learn or do.
    - If it's a job, give a general overview of the responsibilities.
    Keep the explanation concise and focused.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful career counselor AI providing detailed explanations."},
                {"role": "user", "content": prompt}
            ]
        )
        return {"details": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating details: {str(e)}")