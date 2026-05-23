"use client";

import { AuthorDetailResponse } from "@/lib/types";
import { useStore } from "@/lib/store";

interface Props {
  details: AuthorDetailResponse;
}

export default function AuthorDetails({ details }: Props) {
  const setSelectedNodeId = useStore((state) => state.setSelectedNodeId);

  const typeIcon: Record<string, string> = {
    paper: "📄",
    note: "📝",
    book: "📚",
    video: "🎥",
    webpage: "🌐"
  };

  return (
    <div className="details-row">
      <div className="details-card">
        <h3>
          <span className="details-badge badge-author">Автор</span>
          {details.name}
        </h3>
        <div className="details-row" style={{ marginTop: "10px" }}>
          <div className="details-field">
            <div className="details-label">Биография / Описание</div>
            <div className="details-value" style={{ fontSize: "13px", color: "var(--text2)", lineHeight: 1.5 }}>
              Извлечено NER-моделью. Описание отсутствует.
            </div>
          </div>
        </div>
      </div>

      {details.papers && details.papers.length > 0 && (
        <div className="details-card">
          <h3>📚 Работы автора ({details.papers.length})</h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginTop: "10px" }}>
            {details.papers.map((p) => (
              <div
                key={p.id}
                className="details-value"
                style={{ fontSize: "13px", cursor: "pointer", display: "flex", gap: "6px", alignItems: "center" }}
                onClick={() => setSelectedNodeId(p.id)}
              >
                <span>{typeIcon[p.source_type] || "📄"}</span>
                <span style={{ color: "var(--accent)" }}>{p.title}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
