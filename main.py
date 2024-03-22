# Install necessary libraries
# !pip install -U sentence-transformers fastapi uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Sample data
obj = {
    "Who painted the Mona Lisa?": "Leonardo da Vinci painted the Mona Lisa.",
    "What is the Mona Lisa?": "The Mona Lisa is a painting by Leonardo da Vinci.",
    "What is the Mona Lisa's real name?": "The Mona Lisa's real name is La Gioconda.",
    "When was the Mona Lisa painted?": "The Mona Lisa was painted between 1503 and 1506."
}

# Encode questions
tempLst = [model.encode(i) for i in obj]

# Function to compute similarity
def getSimilarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Function to get similar question
def getSimilarQuestion(q1):
    score = [getSimilarity(i, model.encode(q1)) for i in tempLst]
    return obj[list(obj.keys())[score.index(max(score))]]

# Define request body model
class Question(BaseModel):
    question: str

# Define endpoint
@app.post("/get_answer/")
def get_answer(question: Question):
    return {"answer": getSimilarQuestion(question.question)}

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

