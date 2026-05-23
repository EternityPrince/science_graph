"use client";

import { useState, useEffect, useRef } from "react";
import { useStore } from "@/lib/store";
import { parseWikiLinks } from "@/utils/wikiLinks";
import { Marked } from "marked";
import DOMPurify from "dompurify";

interface Message {
  role: "user" | "agent";
  content: string;
  isStreaming?: boolean;
  error?: boolean;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [provider, setProvider] = useState<"local" | "cloud">("local");

  const chatInput = useStore((state) => state.chatInput);
  const setChatInput = useStore((state) => state.setChatInput);
  const graphNodes = useStore((state) => state.graphData?.nodes || []);
  const setSelectedNodeId = useStore((state) => state.setSelectedNodeId);
  const setView = useStore((state) => state.setView);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Sync textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [chatInput]);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const q = chatInput.trim();
    if (!q || isStreaming) return;

    setIsStreaming(true);
    setChatInput("");

    // Append user message
    const userMsg: Message = { role: "user", content: q };
    setMessages((prev) => [...prev, userMsg]);

    // Append empty agent message
    const agentMsg: Message = { role: "agent", content: "", isStreaming: true };
    setMessages((prev) => [...prev, agentMsg]);

    let accumulated = "";

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          limit: 5,
          cloud: provider === "cloud",
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No readable stream in response");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // keep trailing incomplete line

        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (!raw) continue;
          try {
            const ev = JSON.parse(raw);
            if (ev.type === "token") {
              accumulated += ev.text;
              setMessages((prev) => {
                const copy = [...prev];
                const last = copy[copy.length - 1];
                if (last && last.role === "agent") {
                  last.content = accumulated;
                }
                return copy;
              });
            } else if (ev.type === "done") {
              setMessages((prev) => {
                const copy = [...prev];
                const last = copy[copy.length - 1];
                if (last && last.role === "agent") {
                  last.content = accumulated;
                  last.isStreaming = false;
                }
                return copy;
              });
            } else if (ev.type === "error") {
              setMessages((prev) => {
                const copy = [...prev];
                const last = copy[copy.length - 1];
                if (last && last.role === "agent") {
                  last.content = ev.text;
                  last.isStreaming = false;
                  last.error = true;
                }
                return copy;
              });
            }
          } catch (e) {
            // parsing error
          }
        }
      }
    } catch (e: any) {
      setMessages((prev) => {
        const copy = [...prev];
        const last = copy[copy.length - 1];
        if (last && last.role === "agent") {
          last.content = `Ошибка: ${e.message}`;
          last.isStreaming = false;
          last.error = true;
        }
        return copy;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleMessageClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement;
    if (target.classList.contains("wiki-link")) {
      e.preventDefault();
      const nodeId = target.getAttribute("data-node-id");
      if (nodeId) {
        setSelectedNodeId(nodeId);
        setView("graph");
      }
    }
  };

  const renderMessageContent = (msg: Message) => {
    if (msg.error) {
      return `<span style="color:var(--red)">⚠️ ${msg.content}</span>`;
    }
    
    // Parse Markdown securely
    const marked = new Marked();
    const parsed = marked.parse(msg.content || "") as string;
    const sanitized = DOMPurify.sanitize(parsed);
    
    // Parse WikiLinks
    const withWiki = parseWikiLinks(sanitized, graphNodes);
    
    if (msg.isStreaming) {
      return withWiki + '<span class="typing-cursor"></span>';
    }
    
    return withWiki || '<span style="color:var(--text3)">Нет ответа от модели.</span>';
  };

  return (
    <div id="view-chat" className="main-view active">
      <div className="chat-container">
        <div id="chat-messages" onClick={handleMessageClick}>
          {messages.length === 0 ? (
            <div className="chat-welcome">
              <div className="icon">🤖</div>
              <h3>Исследовательский ассистент</h3>
              <p>Задайте вопрос по проиндексированным материалам — статьям, книгам, заметкам.</p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div key={index} className={`msg msg-${msg.role}`}>
                <div className="msg-avatar">{msg.role === "user" ? "👤" : "🤖"}</div>
                <div className="msg-body">
                  <div className="msg-name">{msg.role === "user" ? "Вы" : "Ассистент"}</div>
                  <div 
                    className="msg-text"
                    dangerouslySetInnerHTML={{ __html: renderMessageContent(msg) }}
                  />
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <div className="chat-input-wrap">
          <select 
            value={provider}
            onChange={(e) => setProvider(e.target.value as any)}
            className="settings-select" 
            style={{ width: "auto", maxWidth: "140px", fontSize: "12px", background: "var(--bg)", border: "1px solid var(--border-solid)", color: "var(--text)", borderRadius: "4px", padding: "0 8px", cursor: "pointer", outline: "none" }}
          >
            <option value="local">🤖 Локальный (MLX)</option>
            <option value="cloud">☁️ Облачный ИИ</option>
          </select>
          
          <textarea 
            ref={textareaRef}
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ваш вопрос…" 
            rows={1}
          />
          
          <button 
            id="chat-send" 
            onClick={handleSend}
            disabled={isStreaming || !chatInput.trim()} 
            title="Отправить (Enter)"
          >
            ➤
          </button>
        </div>
      </div>
    </div>
  );
}
