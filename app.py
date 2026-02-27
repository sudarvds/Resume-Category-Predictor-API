from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="AI Resume Category Predictor")

model = joblib.load("resume_model.pkl")

class ResumeInput(BaseModel):
    resume_text: str

@app.get("/")
def home():
    return {"message": "Resume Prediction API Running"}

@app.post("/predict")
def predict(data: ResumeInput):
    prediction = model.predict([data.resume_text])[0]
    probability = max(model.predict_proba([data.resume_text])[0])

    return {
        "predicted_category": prediction,
        "confidence_score": round(float(probability), 3)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
