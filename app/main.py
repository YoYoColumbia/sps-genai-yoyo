from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
import spacy

app = FastAPI()

nlp = spacy.load("en_core_web_sm")

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

class EmbedRequest(BaseModel):
    word: str 

class EmbedResponse(BaseModel):
    word: str
    dim: int
    vector: list[float]
    norm: float

@app.post("/embed", response_model=EmbedResponse)

def embed(request: EmbedRequest):
    doc = nlp(request.word)
    tok = doc[0]
    vector = tok.vector
    norm = tok.vector_norm
    return EmbedResponse(
        word=tok.text,
        dim=len(vector),
        vector=vector.tolist(),
        norm=float(norm)
    )

# RNN Text Generator
from app.rnn_model import RNNTextGenerator
import torch

# Load trained RNN model (just like CNN)
RNN_MODEL_PATH = "app/model_rnn.pth"
RNN_VOCAB_PATH = "app/rnn_vocab.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    rnn_vocab = torch.load(RNN_VOCAB_PATH)
    word2idx = rnn_vocab["word2idx"]
    idx2word = rnn_vocab["idx2word"]

    rnn_model = RNNTextGenerator(vocab_size=len(word2idx)).to(device)
    rnn_model.load_state_dict(torch.load(RNN_MODEL_PATH, map_location=device))
    rnn_model.eval()
    rnn_load_error = None
except Exception as e:
    rnn_model = None
    word2idx = None
    idx2word = None
    rnn_load_error = e

@app.get("/rnn/health")
def rnn_health():
    if rnn_model is None:
        return {"status": "error", "detail": str(rnn_load_error)}
    return {"status": "ok", "device": str(device), "vocab_size": len(word2idx)}

@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    if rnn_model is None:
        return {"error": "RNN model not loaded", "detail": str(rnn_load_error)}

    generated_text = rnn_model.generate(
        start_word=request.start_word.lower(),
        length=request.length,
        word2idx=word2idx,
        idx2word=idx2word,
        device=device
    )
    return {"generated_text": generated_text}

# CNN Image Classifier (CIFAR-10)

import io
import torch
import numpy as np
from PIL import Image
from fastapi import UploadFile, File, HTTPException
from torchvision import transforms
from app.cnn_model import TinyCNN

# Preprocessing (must match your training)
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

# Load trained model + class labels
MODEL_PATH = "model.pth"
CLASSES_PATH = "classes.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = TinyCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    classes = torch.load(CLASSES_PATH)
    model_load_error = None
except Exception as e:
    model = None
    classes = None
    model_load_error = e

@app.get("/clf/health")
def clf_health():
    if model is None or classes is None:
        return {"status": "error", "detail": f"Model not loaded. Error: {str(model_load_error)}"}
    return {"status": "ok", "device": str(device), "num_classes": len(classes)}

@app.get("/clf/labels")
def clf_labels():
    if classes is None:
        raise HTTPException(status_code=503, detail="Classes not loaded. Train first.")
    return {"labels": classes}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or classes is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded. Train first. Error: {str(model_load_error)}")

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return {
        "pred_class": classes[idx],
        "pred_index": idx,
        "confidence": float(probs[idx])
    }
