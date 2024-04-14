import nltk
from nltk.tokenize import sent_tokenize
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import json
from transformers import pipeline
import asyncio
import pymongo

# MongoDB connection details
MONGODB_URL = "mongodb+srv://prkskrs:1JRRLP0TScJtklaB@cluster0.fncdhdb.mongodb.net/myPrjmtDB?retryWrites=true&w=majority"
DB_NAME = "siddagangaDB"
USER_COLLECTION = "users"

# Connect to MongoDB
client = pymongo.MongoClient(MONGODB_URL)
db = client[DB_NAME]
user_collection = db[USER_COLLECTION]


# Download NLTK data for sentence tokenization
nltk.download('punkt')

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

# Initialize summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Sample data
with open("academics.json", "r") as file:
    data = json.load(file)  # Parse JSON data

# Function to encode and compute similarity
def encode_and_compute_similarity(sentence1, sentence2):
    embeddings = model.encode([sentence1, sentence2])
    return np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))

# Function to merge similar answers and rephrase them
def merge_and_rephrase(similar_answers):
    # Merge similar answers
    merged_answer = " ".join(similar_answers)
    
    # Tokenize merged answer into sentences
    sentences = sent_tokenize(merged_answer)
    
    # Simple rephrasing - capitalize the first letter of every sentence
    rephrased_answer = ". ".join([sentence.capitalize() for sentence in sentences])
    
    return rephrased_answer

# Function to get similar questions and merge/rephrase answers
def get_similar_question_and_answer(question):
    similarities = [(q, encode_and_compute_similarity(question, q)) for q in data.keys()]
    max_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:2]  # Get top 5 similar questions
    similar_questions = [sim[0] for sim in max_similarities]
    similar_answers = [data[q] for q in similar_questions]
    
    return similar_questions, similar_answers


# User model for signup
class UserSignup(BaseModel):
    email: str
    password: str

# User model for login
class UserLogin(BaseModel):
    email: str
    password: str

# Define request body model
class Question(BaseModel):
    question: str

# Define endpoint
@app.post("/get_answer/")
async def get_answer(question: Question):
    similar_questions, similar_answers = get_similar_question_and_answer(question.question)
    merged_and_rephrased_answer = merge_and_rephrase(similar_answers)
    
    # Asynchronously summarize merged and rephrased answer
    loop = asyncio.get_event_loop()
    summary_task = loop.run_in_executor(None, summarizer, merged_and_rephrased_answer, 130, 30, False)
    summary = await summary_task
    summarized_text = summary[0]['summary_text']
    
    return {"similar_questions": similar_questions, "merged_and_rephrased_answer": merged_and_rephrased_answer, "answer": summarized_text}

@app.post("/signup/")
async def signup(user: UserSignup):
    # Check if user already exists
    print(user)
    if user_collection.find_one({"email": user.email}):
        return {"message": "User already exists"}

    # Store user in the database
    user_collection.insert_one({"email": user.email, "password": user.password})
    return {"message": "User created successfully"}

# Login endpoint
@app.post("/login/")
async def login(user: UserLogin):
    # Retrieve user from the database
    stored_user = user_collection.find_one({"email": user.email})

    # Check if user exists and verify password
    if user.password == stored_user["password"]:
        return {"message": "Login successful"}
    else:
        return {"message": "Invalid username or password"}

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
