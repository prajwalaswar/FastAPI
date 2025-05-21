from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import time

app = FastAPI(
    title="AI Text Analysis API",
    description="A simple API for text analysis using Hugging Face models",
    version="0.1.0"
)

# Load models
sentiment_analyzer = pipeline("sentiment-analysis")
try:
    text_generator = pipeline("text-generation", model="gpt2", max_length=50)
except:
    text_generator = None

try:
    summarizer = pipeline("summarization")
except:
    summarizer = None

try:
    text_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
except:
    text_classifier = None

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to my AI API", "docs": "/docs"}

@app.post("/sentiment")
def analyze_sentiment(input_data: TextInput):
    try:
        result = sentiment_analyzer(input_data.text)[0]
        return {
            "text": input_data.text,
            "sentiment": result["label"],
            "confidence": result["score"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
def generate_text(input_data: TextInput):
    if not text_generator:
        raise HTTPException(status_code=503, detail="Text generation model not available")
    try:
        result = text_generator(input_data.text)[0]
        return {"prompt": input_data.text, "generated_text": result["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
def classify_text(input_data: TextInput):
    if not text_classifier:
        raise HTTPException(status_code=503, detail="Text classification model not available")
    try:
        result = text_classifier(input_data.text)[0]
        return {
            "text": input_data.text,
            "category": result["label"],
            "confidence": result["score"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
def summarize_text(input_data: TextInput):
    if not summarizer:
        raise HTTPException(status_code=503, detail="Summarization model not available")
    try:
        # For summarization, text should be at least a paragraph
        if len(input_data.text.split()) < 30:
            return {"summary": input_data.text, "note": "Text too short for summarization"}

        result = summarizer(input_data.text, max_length=130, min_length=30, do_sample=False)[0]
        return {"original_text": input_data.text, "summary": result["summary_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

