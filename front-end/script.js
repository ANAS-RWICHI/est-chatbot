document.getElementById("send-btn").addEventListener("click", sendMessage);
document
  .getElementById("user-input")
  .addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      sendMessage();
    }
  });

function sendMessage() {
  const userInput = document.getElementById("user-input");
  const message = userInput.value.trim();
  if (message === "") return;

  console.log("Sending message:", message); // Debug log

  appendMessage("user", message);
  userInput.value = "";

  fetch("http://localhost:5000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json"
    },
    body: JSON.stringify({ message: message }),
  })
    .then((response) => {
      console.log("Response status:", response.status); // Debug log
      return response.json();
    })
    .then((data) => {
      console.log("Received data:", data); // Debug log
      appendMessage("bot", data.response);
    })
    .catch((error) => {
      console.error("Error details:", error); // More detailed error
      appendMessage(
        "bot",
        "Sorry, there was an error processing your request."
      );
    });
}

function appendMessage(sender, message) {
  const chatBox = document.getElementById("chat-box");
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("chat-message");
  messageDiv.classList.add(sender === "user" ? "user-message" : "bot-message");

  // Use innerHTML instead of textContent to render HTML links
  messageDiv.innerHTML = message;

  chatBox.appendChild(messageDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}
