import { postQuery } from './api.js';
import { toast } from './ui.js';
import { escapeHtml } from './utils.js';
import { parseWikiLinks } from './notes.js';

let isStreaming = false;
let viewSwitcherCallback = null;

/**
 * Register view switcher callback.
 */
export function registerViewSwitcher(cb) {
  viewSwitcherCallback = cb;
}

/**
 * Add a message bubble to the chat container.
 */
export function addMessage(role, content = '') {
  const chatWelcome = document.getElementById('chat-welcome');
  if (chatWelcome) chatWelcome.style.display = 'none';

  const chatMessages = document.getElementById('chat-messages');
  if (!chatMessages) return null;

  const div = document.createElement('div');
  div.className = `msg msg-${role}`;
  const avatar = role === 'user' ? '👤' : '🤖';
  const name   = role === 'user' ? 'Вы' : 'Ассистент';
  div.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-body">
      <div class="msg-name">${name}</div>
      <div class="msg-text" id="msg-${Date.now()}"></div>
    </div>`;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  const textBubble = div.querySelector('.msg-text');
  if (content && textBubble) {
    textBubble.textContent = content;
  }
  return textBubble;
}

/**
 * Send user query to RAG and stream answer.
 */
export async function sendMessage() {
  const chatInput = document.getElementById('chat-input');
  const chatSend = document.getElementById('chat-send');
  const chatMessages = document.getElementById('chat-messages');

  if (!chatInput || !chatSend || !chatMessages) return;

  const q = chatInput.value.trim();
  if (!q || isStreaming) return;

  isStreaming = true;
  chatSend.disabled = true;
  chatInput.value = '';
  chatInput.style.height = 'auto';

  // Add user bubble
  addMessage('user', q);

  // Add agent bubble
  const agentBubble = addMessage('agent');
  if (!agentBubble) return;
  agentBubble.innerHTML = '<span class="typing-cursor"></span>';

  let accumulated = '';

  try {
    const response = await postQuery(q, 5);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop(); // Incomplete line goes back to buffer

      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const raw = line.slice(5).trim();
        if (!raw) continue;
        try {
          const ev = JSON.parse(raw);
          if (ev.type === 'token') {
            accumulated += ev.text;
            const parsedHtml = marked.parse(accumulated);
            const sanitizedHtml = DOMPurify.sanitize(parsedHtml);
            agentBubble.innerHTML = parseWikiLinks(sanitizedHtml) + '<span class="typing-cursor"></span>';
            chatMessages.scrollTop = chatMessages.scrollHeight;
          } else if (ev.type === 'done') {
            const parsedHtml = marked.parse(accumulated);
            const sanitizedHtml = DOMPurify.sanitize(parsedHtml);
            agentBubble.innerHTML = parseWikiLinks(sanitizedHtml);
          } else if (ev.type === 'error') {
            agentBubble.innerHTML = `<span style="color:var(--red)">⚠️ ${escapeHtml(ev.text)}</span>`;
          }
        } catch (e) {
          // Silent JSON parse errors for formatting/blank tokens
        }
      }
    }

    if (!accumulated) {
      agentBubble.innerHTML = '<span style="color:var(--text3)">Нет ответа от модели.</span>';
    }

  } catch (e) {
    agentBubble.innerHTML = `<span style="color:var(--red)">⚠️ Ошибка: ${escapeHtml(e.message)}</span>`;
  } finally {
    isStreaming = false;
    chatSend.disabled = false;
  }
}

/**
 * Focuses on the chat, sets a predefined query prompt.
 */
export function askAbout(title) {
  if (viewSwitcherCallback) {
    viewSwitcherCallback('chat');
  }
  const input = document.getElementById('chat-input');
  if (input) {
    input.value = `Расскажи подробнее о работе "${title}"`;
    input.focus();
    // Trigger auto-resize:
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  }
}

// Bind to window for dynamic HTML clicks
window.askAbout = askAbout;
