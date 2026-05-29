"use client";

import { useRouter } from "next/navigation";
import { PaperDetailResponse } from "@/lib/types";
import { useStore } from "@/lib/store";
import { parseWikiLinks } from "@/utils/wikiLinks";
import { openLocalFile } from "@/lib/api";
import { useState } from "react";

interface Props {
  details: PaperDetailResponse;
}

export default function PaperDetails({ details }: Props) {
  const router = useRouter();
  const setSelectedNodeId = useStore((state) => state.setSelectedNodeId);
  const setView = useStore((state) => state.setView);
  const graphNodes = useStore((state) => state.graphData?.nodes) || [];
  const showReferences = useStore((state) => state.showReferences);
  const setShowReferences = useStore((state) => state.setShowReferences);
  const askAbout = useStore((state) => state.askAbout);

  const [localFileStatus, setLocalFileStatus] = useState<string | null>(null);

  const typeLabel: Record<string, string> = {
    paper: "Статья",
    note: "Заметка",
    book: "Книга",
    webpage: "Веб-страница",
    video: "Видео"
  };

  const badgeClass: Record<string, string> = {
    paper: "badge-paper",
    note: "badge-note",
    book: "badge-book",
    webpage: "badge-webpage",
    video: "badge-video"
  };

  const handleWikiLinkClick = (e: React.MouseEvent<HTMLDivElement>) => {
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

  const processText = (text: string) => {
    return parseWikiLinks(text || "", graphNodes);
  };

  const handleOpenLocalFile = async () => {
    if (!details.file_path) return;
    setLocalFileStatus("⏳ Открытие...");
    try {
      const res = await openLocalFile(details.file_path);
      setLocalFileStatus(`✅ ${res.message}`);
      setTimeout(() => setLocalFileStatus(null), 3000);
    } catch (e: any) {
      setLocalFileStatus(`❌ Ошибка: ${e.message}`);
      setTimeout(() => setLocalFileStatus(null), 4000);
    }
  };

  return (
    <div className="details-row">
      <div className="details-card">
        <h3>
          <span className={`details-badge ${badgeClass[details.source_type] || "badge-paper"}`}>
            {typeLabel[details.source_type] || "Документ"}
          </span>
          {details.title}
        </h3>
        <div className="details-row" style={{ marginTop: "10px" }}>
          {details.authors && details.authors.length > 0 && (
            <div className="details-field">
              <div className="details-label">Авторы</div>
              <div className="details-value">{details.authors.join(", ")}</div>
            </div>
          )}
          {details.year && (
            <div className="details-field">
              <div className="details-label">Год</div>
              <div className="details-value">{details.year}</div>
            </div>
          )}
          {details.doi && (
            <div className="details-field">
              <div className="details-label">DOI</div>
              <div className="details-value">
                <a href={`https://doi.org/${details.doi}`} target="_blank" rel="noreferrer">
                  {details.doi}
                </a>
              </div>
            </div>
          )}
          {details.created_at && (
            <div className="details-field">
              <div className="details-label">Добавлен</div>
              <div className="details-value">
                {details.created_at.substring(0, 16).replace("T", " ")}
              </div>
            </div>
          )}
        </div>
      </div>

      {details.abstract && (
        <div className="details-card">
          <h3>📄 Аннотация</h3>
          <div 
            className="abstract-text" 
            style={{ marginTop: "10px" }}
            onClick={handleWikiLinkClick}
            dangerouslySetInnerHTML={{ __html: processText(details.abstract) }}
          />
        </div>
      )}

      {details.summary && (
        <div className="details-card">
          <h3>💡 Краткое содержание (LLM Summary)</h3>
          <div 
            style={{ fontSize: "13px", color: "var(--text2)", lineHeight: "1.6", marginTop: "10px", maxHeight: "250px", overflowY: "auto", background: "var(--surface3)", padding: "10px 12px" }}
            onClick={handleWikiLinkClick}
            dangerouslySetInnerHTML={{ __html: processText(details.summary) }}
          />
          <div style={{ fontSize: "10px", color: "var(--text3)", marginTop: "12px", borderTop: "1px solid var(--border-solid)", paddingTop: "8px" }}>
            🤖 Сгенерировано LLM
          </div>
        </div>
      )}

      {details.concepts && details.concepts.length > 0 && (
        <div className="details-card">
          <h3>🧠 Концепты</h3>
          <div className="tag-list" style={{ marginTop: "10px" }}>
            {details.concepts.map((c) => (
              <span key={c.id} className="tag tag-concept" onClick={() => setSelectedNodeId(c.id)}>
                {c.name}
              </span>
            ))}
          </div>
        </div>
      )}

      {details.tags && details.tags.length > 0 && (
        <div className="details-card">
          <h3>🏷️ Теги</h3>
          <div className="tag-list" style={{ marginTop: "10px" }}>
            {details.tags.map((t) => (
              <span key={t.id} className="tag tag-tag" onClick={() => setSelectedNodeId(t.id)}>
                {t.name}
              </span>
            ))}
          </div>
        </div>
      )}

      {details.citations && details.citations.length > 0 && (
        <div className="details-card">
          <h3>📎 Цитирует ({details.citations.length})</h3>
          <div className="details-row" style={{ marginTop: "10px", gap: "8px" }}>
            {details.citations.map((t) => (
              <div 
                key={t.id} 
                className="details-value" 
                style={{ fontSize: "12px", color: "var(--accent)", cursor: "pointer" }}
                onClick={() => setSelectedNodeId(t.id)}
              >
                • {t.title}
              </div>
            ))}
          </div>
        </div>
      )}

      {details.cited_by && details.cited_by.length > 0 && (
        <div className="details-card">
          <h3>📌 Цитируется в ({details.cited_by.length})</h3>
          <div className="details-row" style={{ marginTop: "10px", gap: "8px" }}>
            {details.cited_by.map((t) => (
              <div 
                key={t.id} 
                className="details-value" 
                style={{ fontSize: "12px", color: "var(--accent)", cursor: "pointer" }}
                onClick={() => setSelectedNodeId(t.id)}
              >
                • {t.title}
              </div>
            ))}
          </div>
        </div>
      )}

      {(details.citations?.length > 0 || details.cited_by?.length > 0) && (
        <button 
          className={`btn ${showReferences ? "btn-danger" : "btn-ghost"}`} 
          style={{ width: "100%", marginTop: "8px", display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}
          onClick={() => setShowReferences(!showReferences)}
        >
          {showReferences ? "❌ Скрыть упомянутые работы с графа" : "🔍 Показать упомянутые работы на графе"}
        </button>
      )}

      {details.file_path && (
        <button 
          className="btn btn-ghost" 
          style={{ width: "100%", marginTop: "8px", display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}
          onClick={handleOpenLocalFile}
        >
          📂 {localFileStatus || "Открыть локальный файл"}
        </button>
      )}

      <button 
        className="btn btn-ghost" 
        style={{ width: "100%", marginTop: "8px", display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}
        onClick={() => router.push("/library?id=" + details.id)}
      >
        📚 Показать в библиотеке
      </button>

      <button 
        className="btn btn-primary" 
        style={{ width: "100%", marginTop: "8px" }}
        onClick={() => askAbout(details.title)}
      >
        💬 Спросить об этой работе
      </button>
    </div>
  );
}
