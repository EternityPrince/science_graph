"use client";

import { useState } from "react";
import { useStore } from "@/lib/store";

export default function DashboardPanel() {
  const {
    stats,
    notes,
    graphData,
    setView,
    setSelectedNodeId,
    setHeatmapDate,
    setChatInput
  } = useStore();

  const [quickQuery, setQuickQuery] = useState("");

  const handleQuickAsk = (e: React.FormEvent) => {
    e.preventDefault();
    const q = quickQuery.trim();
    if (!q) return;
    setChatInput(q);
    setQuickQuery("");
    setView("chat");
  };

  // 1. Calculate degrees for popular concepts & tags
  const edgeCounts: Record<string, number> = {};
  if (graphData && graphData.edges) {
    graphData.edges.forEach((e) => {
      edgeCounts[e.from] = (edgeCounts[e.from] || 0) + 1;
      edgeCounts[e.to] = (edgeCounts[e.to] || 0) + 1;
    });
  }

  const allConceptNodes = graphData?.nodes.filter(
    (n) => n.group === "concept" || n.group === "tag"
  ) || [];

  // Sort concepts by connection degree descending
  const sortedConcepts = [...allConceptNodes]
    .map((c) => ({ ...c, degree: edgeCounts[c.id] || 0 }))
    .sort((a, b) => b.degree - a.degree)
    .slice(0, 15);

  // 2. Filter recent documents
  const allDocNodes = graphData?.nodes.filter(
    (n) => ["paper", "book", "video", "webpage"].includes(n.group)
  ) || [];

  const recentDocs = [...allDocNodes]
    .sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""))
    .slice(0, 5);

  const recentNotes = [...notes]
    .sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""))
    .slice(0, 5);

  // 3. Mini Activity Heatmap
  const dateCounts: Record<string, number> = {};
  graphData?.nodes.forEach((n) => {
    if (n.created_at) {
      const dt = n.created_at.substring(0, 10);
      dateCounts[dt] = (dateCounts[dt] || 0) + 1;
    }
  });

  const cells = [];
  const today = new Date();
  const start = new Date(today);
  // Bento heatmap displays past 12 weeks for compactness (84 cells)
  start.setDate(today.getDate() - 84);

  for (let i = 0; i <= 84; i++) {
    const d = new Date(start);
    d.setDate(start.getDate() + i);
    const dateStr = d.toISOString().substring(0, 10);
    const count = dateCounts[dateStr] || 0;

    let level = 0;
    if (count === 1) level = 1;
    else if (count === 2) level = 2;
    else if (count === 3) level = 3;
    else if (count >= 4) level = 4;

    const formattedDate = d.toLocaleDateString("ru-RU", { day: "numeric", month: "short", year: "numeric" });
    const tooltipText = `${formattedDate}: ${count} док.`;

    cells.push({ dateStr, count, level, tooltipText });
  }

  const handleCellClick = (dateStr: string) => {
    setHeatmapDate(dateStr);
    setView("chronology");
  };

  const typeIcon: Record<string, string> = {
    paper: "📄",
    note: "📝",
    book: "📚",
    video: "🎥",
    webpage: "🌐"
  };

  const formattedStorage = () => {
    if (stats?.storage && stats.storage.total_size !== undefined) {
      const sizeMb = stats.storage.total_size / (1024 * 1024);
      return sizeMb < 0.1 ? "<0.1 MB" : `${sizeMb.toFixed(1)} MB`;
    }
    return "—";
  };

  return (
    <div id="view-dashboard" className="main-view active">
      <div className="dashboard-header">
        <h1>Science Graph</h1>
        <p>Персональная база научных знаний и связей на основе локального искусственного интеллекта</p>
      </div>

      <div className="bento-grid">
        {/* Card 1: Stats */}
        <div className="bento-card bento-stats">
          <h3>📊 Статистика базы</h3>
          <div className="stats-bento-grid">
            <div className="stat-bento-item">
              <span className="stat-bento-num">{stats?.indexed_papers ?? "—"}</span>
              <span className="stat-bento-label">Индексировано</span>
            </div>
            <div className="stat-bento-item">
              <span className="stat-bento-num">{stats?.mentioned_papers ?? "—"}</span>
              <span className="stat-bento-label">Упомянуто</span>
            </div>
            <div className="stat-bento-item">
              <span className="stat-bento-num">{stats?.concepts ?? "—"}</span>
              <span className="stat-bento-label">Концептов</span>
            </div>
            <div className="stat-bento-item">
              <span className="stat-bento-num">{notes.length}</span>
              <span className="stat-bento-label">Заметок</span>
            </div>
            <div className="stat-bento-item">
              <span className="stat-bento-num">{stats?.edges ?? "—"}</span>
              <span className="stat-bento-label">Связей</span>
            </div>
            <div className="stat-bento-item">
              <span className="stat-bento-num">{formattedStorage()}</span>
              <span className="stat-bento-label">Размер</span>
            </div>
          </div>
        </div>

        {/* Card 2: Quick RAG Chat */}
        <div className="bento-card bento-ask">
          <h3>🤖 Быстрый вопрос к ИИ</h3>
          <div className="quick-ask-wrap">
            <p style={{ fontSize: "13px", color: "var(--text2)", marginBottom: "8px" }}>
              Спросите локального ассистента о ваших проиндексированных материалах:
            </p>
            <form onSubmit={handleQuickAsk} className="quick-ask-input-wrap">
              <input
                type="text"
                value={quickQuery}
                onChange={(e) => setQuickQuery(e.target.value)}
                placeholder="Например: Какая архитектура используется в Transformer?"
                autoComplete="off"
              />
              <button type="submit" className="btn btn-primary">Спросить</button>
            </form>
          </div>
        </div>

        {/* Card 3: Activity Heatmap */}
        <div className="bento-card bento-heatmap">
          <h3>📅 Календарь активности</h3>
          <div className="heatmap-grid">
            {cells.map((cell, idx) => (
              <div
                key={idx}
                className={`heatmap-cell level-${cell.level}`}
                title={cell.tooltipText}
                onClick={() => handleCellClick(cell.dateStr)}
              />
            ))}
          </div>
          <div className="heatmap-legend">
            <span>Меньше</span>
            <div className="legend-cell level-0"></div>
            <div className="legend-cell level-1"></div>
            <div className="legend-cell level-2"></div>
            <div className="legend-cell level-3"></div>
            <div className="legend-cell level-4"></div>
            <span>Больше</span>
          </div>
        </div>

        {/* Card 4: Recent documents */}
        <div className="bento-card bento-recent-docs">
          <h3>📄 Последние публикации</h3>
          <div className="bento-list">
            {recentDocs.length === 0 ? (
              <div style={{ fontSize: "12px", color: "var(--text3)", padding: "10px", textAlign: "center" }}>
                Нет публикаций
              </div>
            ) : (
              recentDocs.map((doc) => (
                <div
                  key={doc.id}
                  className="bento-list-item"
                  onClick={() => {
                    setSelectedNodeId(doc.id);
                    setView("graph");
                  }}
                >
                  <span className="bento-list-item-icon">{typeIcon[doc.group] || "📄"}</span>
                  <div className="bento-list-item-content">
                    <div className="bento-list-item-title">{doc.full_title || doc.label}</div>
                    <div className="bento-list-item-meta">
                      {doc.created_at?.substring(0, 10) || ""}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Card 5: Recent notes */}
        <div className="bento-card bento-recent-notes">
          <h3>📝 Недавние заметки</h3>
          <div className="bento-list">
            {recentNotes.length === 0 ? (
              <div style={{ fontSize: "12px", color: "var(--text3)", padding: "10px", textAlign: "center" }}>
                Нет заметок
              </div>
            ) : (
              recentNotes.map((note) => (
                <div
                  key={note.id}
                  className="bento-list-item"
                  onClick={() => {
                    setSelectedNodeId(note.id);
                    setView("graph");
                  }}
                >
                  <span className="bento-list-item-icon">📝</span>
                  <div className="bento-list-item-content">
                    <div className="bento-list-item-title">{note.title}</div>
                    <div className="bento-list-item-meta">
                      {note.created_at?.substring(0, 10) || ""}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Card 6: Top Concepts */}
        <div className="bento-card bento-top-concepts">
          <h3>🧬 Популярные концепты и теги</h3>
          <div className="top-concepts-flex">
            {sortedConcepts.length === 0 ? (
              <div style={{ fontSize: "12px", color: "var(--text3)", padding: "10px" }}>
                Концепты отсутствуют
              </div>
            ) : (
              sortedConcepts.map((concept) => (
                <span
                  key={concept.id}
                  className="concept-tag"
                  onClick={() => {
                    setSelectedNodeId(concept.id);
                    setView("graph");
                  }}
                >
                  {concept.full_title || concept.label}
                  <span className="concept-tag-count">{concept.degree}</span>
                </span>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
