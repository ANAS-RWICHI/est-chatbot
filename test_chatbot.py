import torch
from transformers import CamembertTokenizer, CamembertModel
import torch.nn as nn
import json
from sklearn.metrics import classification_report

# === Load intent data ===
with open("./intent.json", "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

# === Prepare tags and index mappings ===
tags = [intent["tag"] for intent in intents]
tag2idx = {tag: idx for idx, tag in enumerate(tags)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

# === Load tokenizer and model class ===
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

class CamemBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CamemBERTClassifier, self).__init__()
        self.bert = CamembertModel.from_pretrained("camembert-base")
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.last_hidden_state[:, 0])

# === Load trained model ===
model = CamemBERTClassifier(num_classes=len(tags))
model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device("cpu")))
model.eval()

# === Inference function ===
def predict_intent(user_input):
    user_input = user_input.lower()
    with torch.no_grad():
        encoding = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        outputs = model(encoding["input_ids"], encoding["attention_mask"])
        predicted = torch.argmax(outputs, dim=1).item()
        return idx2tag[predicted]

# === Test Cases ===
test_cases = [
    {"input": "Bonjour", "expected_tag": "greeting"},
    {"input": "Comment s'inscrire à l'EST ?", "expected_tag": "admission_2024"},
    {"input": "Quels sont les DUT proposés ?", "expected_tag": "formations_dut"},
    {"input": "Y a-t-il un programme Erasmus ?", "expected_tag": "international"},
    {"input": "Comment accéder à Moodle ?", "expected_tag": "plateforme_elearning"},
    {"input": "Je veux rejoindre un club", "expected_tag": "vie_etudiante"},
    {"input": "C’est quoi l’EST Fès ?", "expected_tag": "presentation_est"},
    {"input": "Je veux rejoindre un club", "expected_tag": "vie_etudiante"},

]

# === Evaluation ===
y_true = []
y_pred = []

print("\n🔍 Starting evaluation...\n")

for case in test_cases:
    predicted_tag = predict_intent(case["input"])
    y_true.append(case["expected_tag"])
    y_pred.append(predicted_tag)
    print(f"🗨️  Input: {case['input']}")
    print(f"✅ Expected: {case['expected_tag']} | 🤖 Predicted: {predicted_tag}")
    print("✔️ Correct\n" if predicted_tag == case["expected_tag"] else "❌ Incorrect\n")

# === Report ===
print("📊 Classification Report:\n")
print(classification_report(y_true, y_pred))
