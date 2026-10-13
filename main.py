import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
model = joblib.load(BASE / "model" / "advisor_model.pkl")
meta = json.loads((BASE / "model" / "model_metadata.json").read_text())
FEATURES = meta["features"]  # keeps column order identical to training

app = FastAPI(title="Academic Advisor System", version="1.0")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


class Student(BaseModel):
    study_hours_per_week: int = Field(..., ge=0, le=80)
    attendance_pct: int = Field(..., ge=0, le=100)
    previous_gpa: float = Field(..., ge=0, le=5)
    assignments_completed_pct: int = Field(..., ge=0, le=100)
    extracurricular_hours: int = Field(..., ge=0, le=40)
    level: Literal["100L", "200L", "300L", "400L"]


def build_advice(s: Student, status: str) -> list[str]:
    tips = []
    if s.attendance_pct < 75:
        tips.append("Raise attendance above 75%; missed lectures are strongly linked to lower grades.")
    if s.study_hours_per_week < 15:
        tips.append("Aim for at least 15-20 focused study hours per week.")
    if s.assignments_completed_pct < 80:
        tips.append("Complete and submit more assignments; they build continuous-assessment marks.")
    if s.extracurricular_hours > 10:
        tips.append("Review your time balance; extracurricular load is high.")
    if status == "At Risk":
        tips.append("Book a meeting with your academic advisor this week.")
    elif status == "High Performer":
        tips.append("Consider research opportunities, mentoring or scholarships.")
    return tips or ["Keep up your current habits."]


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": meta["model_name"], "sklearn_version": meta["sklearn_version"]}


@app.post("/predict")
def predict(student: Student):
    X = pd.DataFrame([student.model_dump()])[FEATURES]
    status = str(model.predict(X)[0])
    probs = {str(c): round(float(p), 3) for c, p in zip(model.classes_, model.predict_proba(X)[0])}
    return {"status": status, "probabilities": probs, "advice": build_advice(student, status)}