# Install necessary libraries
# !pip install -U sentence-transformers fastapi uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import json  # Import the json module

# Initialize FastAPI app
app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Sample data
with open("academics.json", "r") as file:
    obj = json.load(file)  # Parse JSON data


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

