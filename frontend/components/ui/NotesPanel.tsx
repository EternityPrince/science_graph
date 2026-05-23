"use client";

import { useState } from "react";
import { useStore } from "@/lib/store";
import { saveNote } from "@/lib/api";

export default function NotesPanel() {
  const notes = useStore((state) => state.notes);
  const setSelectedNodeId = useStore((state) => state.setSelectedNodeId);
  const setView = useStore((state) => state.setView);
  const refreshAll = useStore((state) => state.refreshAll);

  const [title, setTitle] = useState("");
  const [authorsRaw, setAuthorsRaw] = useState("");
  const [tagsRaw, setTagsRaw] = useState("");
  const [content, setContent] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [statusMsg, setStatusMsg] = useState<{ text: string; type: "ok" | "err" } | null>(null);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !content.trim()) return;

    setIsSaving(true);
    setStatusMsg(null);

    const authors = authorsRaw ? authorsRaw.split(",").map((a) => a.trim()).filter(Boolean) : [];
    const tags = tagsRaw ? tagsRaw.split(",").map((t) => t.trim()).filter(Boolean) : [];

    try {
      await saveNote({ title, content, authors, tags });
      setStatusMsg({ text: "✅ Заметка успешно создана", type: "ok" });
      setTitle("");
      setAuthorsRaw("");
      setTagsRaw("");
      setContent("");
      
      // Refresh database records, graph, and statistics
      await refreshAll();
    } catch (err: any) {
      setStatusMsg({ text: `❌ Ошибка: ${err.message}`, type: "err" });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div id="view-notes" className="main-view active">
      <div className="notes-container">
        {/* Left side: List of notes */}
        <div className="notes-list-pane">
          <div className="details-card" style={{ height: "100%", overflowY: "auto" }}>
            <h3>📋 Мои заметки</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginTop: "12px" }}>
              {notes.length === 0 ? (
                <div style={{ fontSize: "12px", color: "var(--text3)", textAlign: "center", padding: "10px" }}>
                  Заметок пока нет
                </div>
              ) : (
                notes.map((n) => {
                  const dateStr = n.created_at ? n.created_at.substring(0, 10) : "—";
                  return (
                    <div
                      key={n.id}
                      className="details-card"
                      style={{ marginBottom: "8px", padding: "12px", cursor: "pointer", borderColor: "var(--border)" }}
                      onClick={() => {
                        setSelectedNodeId(n.id);
                        setView("graph");
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                        <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--text)" }}>{n.title}</span>
                      </div>
                      <div style={{ fontSize: "11px", color: "var(--text3)", display: "flex", justifyContent: "space-between" }}>
                        <span>{n.authors.join(", ") || "Без автора"}</span>
                        <span>{dateStr}</span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>

        {/* Right side: Create/Edit notes form */}
        <div className="notes-form-pane">
          <div className="details-card">
            <h3>📝 Создать заметку</h3>
            <form onSubmit={handleSave} style={{ marginTop: "12px" }}>
              <div className="form-group">
                <label htmlFor="note-title">Название</label>
                <input
                  type="text"
                  id="note-title"
                  className="form-control"
                  placeholder="Название заметки..."
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="note-authors">Авторы (через запятую)</label>
                <input
                  type="text"
                  id="note-authors"
                  className="form-control"
                  placeholder="Имя Автора, Другой Автор..."
                  value={authorsRaw}
                  onChange={(e) => setAuthorsRaw(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label htmlFor="note-tags">Теги (через запятую)</label>
                <input
                  type="text"
                  id="note-tags"
                  className="form-control"
                  placeholder="tag1, tag2..."
                  value={tagsRaw}
                  onChange={(e) => setTagsRaw(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label htmlFor="note-content">Содержание (поддерживает [[wiki-ссылки]])</label>
                <textarea
                  id="note-content"
                  className="form-control"
                  placeholder="Текст заметки с [[другими статьями]]..."
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  required
                />
              </div>
              <button 
                type="submit" 
                className="btn btn-primary" 
                style={{ width: "100%" }}
                disabled={isSaving}
              >
                {isSaving ? "Сохранение..." : "Сохранить заметку"}
              </button>
              {statusMsg && (
                <div style={{ marginTop: "12px", fontSize: "12px", color: statusMsg.type === "ok" ? "var(--green)" : "var(--red)" }}>
                  {statusMsg.text}
                </div>
              )}
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
