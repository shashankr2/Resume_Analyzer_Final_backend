import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://resumiq.vercel.app"],  # Replace with React origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    # Step 1: Extract text from resume PDF
    try:
        reader = PdfReader(resume.file)
        resume_text = "".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        return {"error": f"Failed to read PDF: {str(e)}"}

    # Step 2: Construct prompt
    prompt = f"""
You are an expert in HR resume screening.

Compare this resume and job description.
Return a JSON with:
- score (0-100)
- strengths (list of 3)
- weaknesses (list of 3)
- missing_keywords (list)
- improvement_tips (list)

RESUME:
{resume_text[:6000]}

JOB DESCRIPTION:
{job_description[:2000]}
"""

    # Step 3: Ask Gemini
    try:
        response = model.generate_content(prompt)
        raw_output = response.text

        # Try to parse Gemini's response safely
        json_start = raw_output.find("{")
        json_end = raw_output.rfind("}") + 1

        json_text = raw_output[json_start:json_end]
        parsed = json.loads(json_text)

        # Validate keys
        expected_keys = ["score", "strengths", "weaknesses", "missing_keywords", "improvement_tips"]
        for key in expected_keys:
            if key not in parsed:
                raise ValueError(f"Missing key in Gemini response: {key}")

        return parsed

    except Exception as e:
        print(f"[ERROR] Gemini returned bad format: {e}")
        print(f"[RAW OUTPUT] {raw_output}")

        # Fallback default result
        return {
            "score": 70,
            "strengths": ["Resume is well structured", "Relevant skills mentioned", "Good formatting"],
            "weaknesses": ["Lacks job-specific achievements", "No mention of leadership", "Missing technical keywords"],
            "missing_keywords": ["Flask", "SQL", "Team management"],
            "improvement_tips": [
                "Add measurable outcomes to past roles",
                "Include relevant tools like Flask or SQL",
                "Customize summary for the job role"
            ],
            "note": "Fallback response used due to parsing error."
        }

