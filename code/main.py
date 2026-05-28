import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 💡 動態把目前檔案所在的 code 資料夾加入 Python 的搜尋路徑中，徹底解決 Import 找不到的問題
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

app = FastAPI()

# 允許前端網頁連線的安全設定 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 💡 改用最單純的「相對路徑」，因為等一下我們會直接在 code 資料夾內執行它
# 透過 abspath 轉換，讓 transformers 明白這是本地系統路徑
from pathlib import Path

# 新的
MODEL_PATH = Path(__file__).parent.parent / "bert_sexism_model"
MODEL_PATH = MODEL_PATH.resolve()

print(f"正在從此路徑載入模型: {MODEL_PATH}")

tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
model = BertForSequenceClassification.from_pretrained(
    str(MODEL_PATH), local_files_only=True
)


class SexismRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(request: SexismRequest):
    inputs = tokenizer(
        request.text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()
    status = "Sexist" if prediction == 1 else "Not Sexist"

    return {"prediction": status, "text": request.text}


# 🚀 終極大招：讓這個檔案可以直接被 python 指令執行
if __name__ == "__main__":
    import uvicorn

    # 強制指定主機為 localhost (127.0.0.1)，連接埠為 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
