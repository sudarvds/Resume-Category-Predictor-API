from fastapi import FastAPI, UploadFile, File
import uvicorn
import pickle
import PyPDF2
from io import BytesIO

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Resume Category Predictor API is Running 🚀"}


# Load your trained model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@app.post("/predict")
async def predict(resume: UploadFile = File(...)):
    
    contents = await resume.read()
    pdf_file = BytesIO(contents)
    
    resume_text = extract_text_from_pdf(pdf_file)

    transformed = vectorizer.transform([resume_text])
    prediction = model.predict(transformed)[0]

    return {
        "predicted_category": prediction
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
