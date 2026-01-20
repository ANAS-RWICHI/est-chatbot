import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizer, CamembertModel
import torch.nn as nn
import torch.optim as optim
import random

# Load intents
with open("intent.json", "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

# Preprocess data
all_patterns = []
all_tags = []
tags = []

for intent in intents:
    tag = intent["tag"]
    if tag not in tags:
        tags.append(tag)
    for pattern in intent["patterns"]:
        all_patterns.append(pattern.lower())  # Lowercasing here
        all_tags.append(tag)

# Encode tags
tag2idx = {tag: idx for idx, tag in enumerate(tags)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}
labels = [tag2idx[tag] for tag in all_tags]

# Tokenizer
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

class IntentDataset(Dataset):
    def __init__(self, patterns, labels):
        self.encodings = tokenizer(patterns, truncation=True, padding=True, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

dataset = IntentDataset(all_patterns, labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
class CamemBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CamemBERTClassifier, self).__init__()
        self.bert = CamembertModel.from_pretrained("camembert-base")
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.last_hidden_state[:, 0])

model = CamemBERTClassifier(num_classes=len(tags))
optimizer = optim.Adam(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()


# Training
model.train()
for epoch in range(5):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Époque {epoch+1}: loss = {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "trained_model.pth")
print("✅ Modèle entraîné et sauvegardé sous 'trained_model.pth'")


# Inference function
model.eval()
def get_response(user_input):
    user_input = user_input.lower()  # Lowercasing user input
    with torch.no_grad():
        encoding = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        outputs = model(encoding["input_ids"], encoding["attention_mask"])
        predicted = torch.argmax(outputs, dim=1).item()
        tag = idx2tag[predicted]
        for intent in intents:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])

# Chat loop
print("🤖 Chatbot EST Fès (CamemBERT) est prêt ! (Tapez 'quit' pour quitter)")
while True:
    msg = input("Vous: ")
    if msg.lower() == "quit":
        break
    response = get_response(msg)
    print("Bot:", response)

