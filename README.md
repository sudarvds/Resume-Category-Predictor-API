# 📄 Resume Category Predictor API

An AI-powered web service built with **FastAPI** that classifies uploaded resumes into specific professional categories (e.g., Data Science, Web Development, HR) using a pre-trained Machine Learning model.

## 🚀 Live Demo
The API is deployed on Render: [Your-Render-URL-Here](https://your-app.onrender.com)

## ✨ Features
- **Fast Inference**: Predicts resume categories in milliseconds.
- **RESTful API**: Simple POST endpoint to send resume text or files.
- **Auto-Documentation**: Interactive Swagger UI available at `/docs`.
- **Pre-trained Model**: Uses a [Random Forest/SVM/etc.] model trained on the Kaggle Resume Dataset.

## 🛠️ Tech Stack
- **Framework**: [FastAPI](https://fastapi.tiangolo.com)
- **Machine Learning**: Scikit-Learn, Pandas, Numpy
- **NLP**: NLTK (for text cleaning & stopwords)
- **Deployment**: Render & GitHub

## 📂 Project Structure
```text
├── main.py              # FastAPI application entry point
├── model.pkl            # Pre-trained ML model file
├── vectorizer.pkl       # TF-IDF or CountVectorizer file
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation

⚙️ Installation & Local Setup
Clone the repository:
bash
git clone https://github.com
cd resume-predictor-api


Create a virtual environment:
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
bash
pip install -r requirements.txt


Run the API:
bash
uvicorn main:app --reload
.

The API will be available at http://127.0.0.1:8000. 
🔌 API Usage
Predict Category
Endpoint: POST /predict
Payload:
json
{
  "resume_text": "Experienced software engineer with skills in Python and FastAPI..."
}
Use code with caution.

Response:
json
{
  "category": "Data Science",
  "confidence": 0.92
}
Use code with caution.

📜 License
This project is licensed under the MIT License - see the LICENSE file for details. 

---




