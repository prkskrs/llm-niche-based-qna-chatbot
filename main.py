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

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
