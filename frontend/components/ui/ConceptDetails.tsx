"use client";

import { ConceptDetailResponse } from "@/lib/types";
import { useStore } from "@/lib/store";

interface Props {
  details: ConceptDetailResponse;
}

export default function ConceptDetails({ details }: Props) {
  const setSelectedNodeId = useStore((state) => state.setSelectedNodeId);

  const typeIcon: Record<string, string> = {
    paper: "📄",
    note: "📝",
    book: "📚",
    video: "🎥",
    webpage: "🌐"
  };

  const isTag = details.type === "tag";
  const badgeClass = isTag ? "badge-tag" : "badge-concept";
  const label = isTag ? "Тег" : "Концепт";

  return (
    <div className="details-row">
      <div className="details-card">
        <h3>
          <span className={`details-badge ${badgeClass}`}>{label}</span>
          {details.name}
        </h3>
        <div className="details-row" style={{ marginTop: "10px" }}>
          <div className="details-field">
            <div className="details-label">Описание</div>
            <div className="details-value" style={{ fontSize: "13px", color: "var(--text2)", lineHeight: 1.5 }}>
              {details.description || "Описание отсутствует."}
            </div>
          </div>
        </div>
      </div>

      {details.papers && details.papers.length > 0 && (
        <div className="details-card">
          <h3>📚 Упоминания в работах ({details.papers.length})</h3>
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

      {details.related && details.related.length > 0 && (
        <div className="details-card">
          <h3>{isTag ? "🧠 Связанные концепты" : "🏷️ Связанные теги"}</h3>
          <div className="tag-list" style={{ marginTop: "10px" }}>
            {details.related.map((item) => (
              <span
                key={item.id}
                className="tag"
                style={{ borderColor: isTag ? undefined : "var(--col-tag)", color: isTag ? undefined : "var(--col-tag)" }}
                onClick={() => setSelectedNodeId(item.id)}
              >
                {item.name}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
