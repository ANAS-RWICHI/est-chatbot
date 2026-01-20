# chatbot_model.py

import torch
import torch.nn as nn
from transformers import CamembertTokenizer, CamembertModel
import json
import random

# Load intents
with open("intent.json", "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

# Prepare tags
tags = [intent["tag"] for intent in intents]
tag2idx = {tag: idx for idx, tag in enumerate(tags)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

# Tokenizer
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

# Model definition
class CamemBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CamemBERTClassifier, self).__init__()
        self.bert = CamembertModel.from_pretrained("camembert-base")
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.last_hidden_state[:, 0])

# Load model
model = CamemBERTClassifier(num_classes=len(tags))
model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device("cpu")))
model.eval()

# Inference
def get_response(user_input):
    user_input = user_input.lower()
    with torch.no_grad():
        encoding = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        outputs = model(encoding["input_ids"], encoding["attention_mask"])
        predicted = torch.argmax(outputs, dim=1).item()
        tag = idx2tag[predicted]
        for intent in intents:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return "Désolé, je n'ai pas compris. Pouvez‑vous reformuler ?"
