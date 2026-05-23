"use client";

import { useState, useRef } from "react";
import { useStore } from "@/lib/store";
import { uploadFile, indexUrl } from "@/lib/api";

interface LogMessage {
  text: string;
  type: "info" | "ok" | "err";
}

export default function UploadPanel() {
  const refreshAll = useStore((state) => state.refreshAll);

  // File Upload states
  const [fileLogs, setFileLogs] = useState<LogMessage[]>([{ text: "Ожидание файла…", type: "info" }]);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // URL Ingest states
  const [urlInput, setUrlInput] = useState("");
  const [urlLogs, setUrlLogs] = useState<LogMessage[]>([{ text: "Ожидание ссылки…", type: "info" }]);
  const [isUrlIngesting, setIsUrlIngesting] = useState(false);

  const addFileLog = (text: string, type: "info" | "ok" | "err") => {
    setFileLogs((prev) => [...prev, { text, type }]);
  };

  const addUrlLog = (text: string, type: "info" | "ok" | "err") => {
    setUrlLogs((prev) => [...prev, { text, type }]);
  };

  const handleFileUpload = async (file: File) => {
    setFileLogs([]);
    addFileLog(`📤 Загрузка: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`, "info");
    addFileLog("⏳ Индексация…", "info");

    try {
      const res = await uploadFile(file);
      addFileLog(`✅ Успешно проиндексировано: ID = ${res.id}`, "ok");
      
      // Refresh app data
      setTimeout(() => {
        refreshAll();
      }, 800);
    } catch (e: any) {
      addFileLog(`❌ Ошибка: ${e.message}`, "err");
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleUrlIngest = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const url = urlInput.trim();
    if (!url || isUrlIngesting) return;

    setIsUrlIngesting(true);
    setUrlLogs([]);
    addUrlLog(`⏳ Индексируем ссылку: ${url}...`, "info");

    try {
      const res = await indexUrl(url);
      addUrlLog(`✅ Успешно проиндексировано: ${res.title || url}`, "ok");
      setUrlInput("");
      
      // Refresh app data
      setTimeout(() => {
        refreshAll();
      }, 800);
    } catch (e: any) {
      addUrlLog(`❌ Ошибка: ${e.message}`, "err");
    } finally {
      setIsUrlIngesting(false);
    }
  };

  return (
    <div id="view-upload" className="main-view active">
      <div className="upload-container">
        {/* File Ingestion Card */}
        <div className="details-card" style={{ width: "100%", marginBottom: 0 }}>
          <h3 style={{ marginBottom: "16px" }}>📤 Загрузка файлов</h3>
          <div 
            className={`drop-zone ${dragOver ? "drag-over" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const file = e.dataTransfer.files[0];
              if (file) handleFileUpload(file);
            }}
          >
            <div className="drop-icon">📂</div>
            <h3>Перетащите файл или кликните</h3>
            <p>Поддерживаются PDF, Markdown (.md) и EPUB</p>
            <input 
              type="file" 
              ref={fileInputRef}
              accept=".pdf,.md,.epub" 
              style={{ display: "none" }} 
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file);
              }}
            />
          </div>
          
          <div className="upload-log" id="upload-log" style={{ marginTop: "16px" }}>
            {fileLogs.map((log, index) => (
              <span key={index} className={`log-${log.type}`}>
                {log.text}
                {"\n"}
              </span>
            ))}
          </div>
        </div>

        {/* URL Ingestion Card */}
        <div className="details-card" style={{ width: "100%", marginBottom: 0 }}>
          <h3 style={{ marginBottom: "16px" }}>🌐 Индексировать URL</h3>
          <form onSubmit={handleUrlIngest} className="url-ingest-wrap">
            <p style={{ fontSize: "13px", color: "var(--text2)", marginBottom: "12px" }}>
              Введите ссылку на YouTube-видео или веб-страницу для добавления в базу знаний:
            </p>
            <div style={{ display: "flex", gap: "8px" }}>
              <input
                type="url"
                className="form-control"
                placeholder="https://www.youtube.com/watch?v=... или https://example.com"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                style={{ flex: 1, background: "var(--surface3)", color: "var(--text)", border: "1px solid var(--border)" }}
                required
              />
              <button 
                type="submit" 
                className="btn btn-primary"
                disabled={isUrlIngesting}
              >
                Индексировать
              </button>
            </div>
          </form>

          <div className="upload-log" id="url-ingest-log" style={{ marginTop: "12px" }}>
            {urlLogs.map((log, index) => (
              <span key={index} className={`log-${log.type}`}>
                {log.text}
                {"\n"}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
