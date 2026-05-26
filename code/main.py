from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

# 初始化模型與 Tokenizer (對應報告中採用的架構) [cite: 12, 35]
MODEL_PATH = "./bert_sexism_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

class SexismRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: SexismRequest):
    # 進行推論
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 取得結果 [cite: 102]
    prediction = torch.argmax(outputs.logits, dim=1).item()
    status = "Sexist" if prediction == 1 else "Not Sexist"
    
    return {"prediction": status, "text": request.text}