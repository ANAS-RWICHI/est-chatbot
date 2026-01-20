# 🤖 EST Fès Chatbot

This is a French-language chatbot designed for students and visitors of **École Supérieure de Technologie de Fès (EST Fès)**. It uses a trained **CamemBERT-based model** for understanding user messages (intent classification) and provides accurate, predefined responses using a Flask API and a simple web frontend.

---

## 🧰 Technologies Used

- 🧠 **CamemBERT** (Hugging Face Transformers)
- 🐍 **Flask** (Python backend API)
- 🌐 **HTML/CSS/JavaScript** (Frontend interface)
- 📦 **PyTorch** (Model training and inference)

---

## 📁 Project Structure

```
est-chatbot/
├── app.py                # Flask API to serve responses
├── chatbot_model.py      # Loads model and handles prediction
├── intent.json           # Dataset of intents and responses
├── trained_model.pth     # Trained CamemBERT model
├── requirements.txt      # Python dependencies
├── frontend/
│   ├── index.html        # Web interface
│   ├── style.css         # Styling for chat UI
│   └── script.js         # JS logic to interact with backend
```

---

## 🚀 How to Run the Project

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/ANAS-RWICHI/est-chatbot.git
cd est-chatbot
```

---

### 🐍 2. Set Up Python Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure Python 3.8+ is installed.

---

### 🔌 3. Start the Flask Backend

```bash
python app.py
```

You should see:

```
Running on http://127.0.0.1:5000/
```

This launches the chatbot backend API at **`http://localhost:5000/chat`**.

---

### 💬 4. Use the Web Chat Interface

Open the frontend:

```bash
cd frontend
```

Open `index.html` in your browser:

- Double-click it, or
- Run `open index.html` (macOS), or
- Use VS Code Live Server / browser plugin

Now, type a message like:

```
Quels DUT proposez-vous ?
```

The bot will reply with structured text and links (e.g., to the EST Fès website).

---

## ✅ Example API Request (Optional)

If you want to test the API directly:

```bash
curl -X POST http://127.0.0.1:5000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Bonjour"}'
```

---

## 👨‍💻 Author

Developed by **[ANAS RWCHI]**  
GitHub: [github.com/ANAS-RWICHI](https://github.com/ANAS-RWICHI)

---

## 📄 License

MIT License — free to use, modify, and share for educational purposes.
