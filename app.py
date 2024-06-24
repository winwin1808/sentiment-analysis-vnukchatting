from typing import List

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertModel, BertTokenizer


# Define the sentiment classifier class
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = bert_output.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

# Function to preprocess input text
def preprocess_input(input_text, tokenizer):
    encoding = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoding['input_ids'], encoding['attention_mask']

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the saved model
model = SentimentClassifier(n_classes=2)
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
model.eval()

# Define FastAPI
app = FastAPI()

# Define input data model
class TextRequest(BaseModel):
    text: str

# Endpoint to predict sentiment
@app.post("/predict")
def predict_sentiment(request: TextRequest):
    text = request.text
    input_ids, attention_mask = preprocess_input(text, tokenizer)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
    predicted_class = torch.argmax(output, dim=1).item()
    sentiment_label = "positive" if predicted_class == 1 else "negative"
    return {"text": text, "sentiment": sentiment_label}

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
