const API_URL = "http://127.0.0.1:8000/chat/stream";

const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");

const initialHTML = chatMessages.innerHTML;

function escapeHTML(str) {
  return str
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatText(text) {
  const escaped = escapeHTML(text);
  return escaped
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
}

function appendMessage(role, content, isError = false) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}${isError ? " error" : ""}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "Tú" : "⚜";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<p>${formatText(content)}</p>`;

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return wrapper;
}

function appendTypingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "⚜";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return wrapper;
}

async function sendMessage(text) {
  const message = text.trim();
  if (!message) return;

  appendMessage("user", message);
  userInput.value = "";
  userInput.style.height = "auto";
  sendBtn.disabled = true;

  const typingEl = appendTypingIndicator();
  let assistantWrapper = null;

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      let detail = "Error en la respuesta del servidor.";
      try {
        const data = await response.json();
        detail = data.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }

    if (!response.body) {
      throw new Error("El navegador no soporta streaming de respuesta.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let fullText = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      if (!chunk) continue;

      if (!assistantWrapper) {
        typingEl.remove();
        assistantWrapper = appendMessage("assistant", "");
      }

      fullText += chunk;
      const bubble = assistantWrapper.querySelector(".bubble");
      bubble.innerHTML = `<p>${formatText(fullText)}</p>`;
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    if (!assistantWrapper) {
      typingEl.remove();
      appendMessage("assistant", "No se recibió respuesta del modelo.");
    }
  } catch (err) {
    typingEl.remove();
    appendMessage(
      "assistant",
      `No se pudo conectar con la API (${err.message}). Verifica que FastAPI esté corriendo en http://127.0.0.1:8000.`,
      true
    );
  } finally {
    sendBtn.disabled = false;
    userInput.focus();
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(userInput.value);
});

userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = `${Math.min(userInput.scrollHeight, 150)}px`;
});

chatMessages.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (chip) {
    sendMessage(chip.dataset.prompt);
  }
});

clearBtn.addEventListener("click", () => {
  chatMessages.innerHTML = initialHTML;
  userInput.focus();
});
