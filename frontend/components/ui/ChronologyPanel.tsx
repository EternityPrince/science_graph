"use client";

import { useStore } from "@/lib/store";

export default function ChronologyPanel() {
  const graphNodes = useStore((state) => state.graphData?.nodes) || [];
  const heatmapDate = useStore((state) => state.heatmapDate);
  const setHeatmapDate = useStore((state) => state.setHeatmapDate);
  const fromDate = useStore((state) => state.fromDate);
  const setFromDate = useStore((state) => state.setFromDate);
  const toDate = useStore((state) => state.toDate);
  const setToDate = useStore((state) => state.setToDate);
  const setSelectedNodeId = useStore((state) => state.setSelectedNodeId);
  const setView = useStore((state) => state.setView);

  // 1. Build date counters from graph nodes
  const dateCounts: Record<string, number> = {};
  graphNodes.forEach((n) => {
    if (n.created_at) {
      const dt = n.created_at.substring(0, 10);
      dateCounts[dt] = (dateCounts[dt] || 0) + 1;
    }
  });

  // 2. Generate cells for the 53 weeks grid
  const cells = [];
  const today = new Date();
  const start = new Date(today);
  start.setDate(today.getDate() - 370);

  for (let i = 0; i <= 370; i++) {
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

  // 3. Handle heatmap cell toggle
  const handleCellClick = (dateStr: string) => {
    if (heatmapDate === dateStr) {
      setHeatmapDate(null);
    } else {
      setHeatmapDate(dateStr);
    }
  };

  const handleResetFilters = () => {
    setHeatmapDate(null);
    setFromDate("");
    setToDate("");
  };

  // 4. Retrieve filtered timeline documents
  const docs = graphNodes.filter((item) => 
    ["paper", "note", "book", "video", "webpage"].includes(item.group)
  );

  // Sort descending by created_at
  docs.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));

  const filteredDocs = docs.filter((n) => {
    if (!n.created_at) return false;
    const docDate = n.created_at.substring(0, 10);
    
    if (heatmapDate) {
      return docDate === heatmapDate;
    }
    if (fromDate && docDate < fromDate) return false;
    if (toDate && docDate > toDate) return false;
    return true;
  });

  const typeIcon: Record<string, string> = {
    paper: "📄",
    note: "📝",
    book: "📚",
    video: "🎥",
    webpage: "🌐"
  };

  return (
    <div id="view-chronology" className="main-view active">
      <div className="chronology-container">
        {/* Left Column: Heatmap and date filter */}
        <div className="chronology-left-pane">
          {/* Heatmap Grid */}
          <div className="heatmap-container">
            <div className="heatmap-header">Календарь активности (53 недели)</div>
            <div className="heatmap-grid" id="heatmap-grid">
              {cells.map((cell, idx) => (
                <div
                  key={idx}
                  className={`heatmap-cell level-${cell.level}`}
                  style={{
                    outline: heatmapDate === cell.dateStr ? "2px solid var(--accent)" : "none",
                    outlineOffset: "-1px"
                  }}
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

          {/* Date Range Filter */}
          <div className="details-card">
            <h3>📅 Фильтр по датам</h3>
            <div style={{ display: "flex", gap: "10px", marginBottom: "10px", marginTop: "12px" }}>
              <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
                <label htmlFor="filter-from-date">С</label>
                <input
                  type="date"
                  id="filter-from-date"
                  className="form-control"
                  value={fromDate}
                  onChange={(e) => {
                    setHeatmapDate(null);
                    setFromDate(e.target.value);
                  }}
                />
              </div>
              <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
                <label htmlFor="filter-to-date">По</label>
                <input
                  type="date"
                  id="filter-to-date"
                  className="form-control"
                  value={toDate}
                  onChange={(e) => {
                    setHeatmapDate(null);
                    setToDate(e.target.value);
                  }}
                />
              </div>
            </div>
            <div style={{ display: "flex", gap: "10px" }}>
              <button className="btn btn-ghost" style={{ flex: 1, fontSize: "11px", padding: "6px" }} onClick={handleResetFilters}>
                Сбросить
              </button>
            </div>
            {heatmapDate && (
              <div id="filter-status-msg" style={{ fontSize: "11px", color: "var(--accent)", marginTop: "8px" }}>
                Выбран день: {heatmapDate}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Timeline */}
        <div className="chronology-right-pane">
          <div className="details-card" style={{ height: "100%", display: "flex", flexDirection: "column", overflow: "hidden", marginBottom: 0 }}>
            <h3>⏳ Хронология публикаций</h3>
            <div id="timeline-list">
              {filteredDocs.length === 0 ? (
                <div style={{ fontSize: "12px", color: "var(--text3)", textAlign: "center", padding: "10px" }}>
                  В выбранном диапазоне нет документов
                </div>
              ) : (
                filteredDocs.map((d) => {
                  const dateStr = d.created_at ? d.created_at.substring(0, 16).replace("T", " ") : "—";
                  const label = d.full_title || d.id;
                  return (
                    <div
                      key={d.id}
                      className="details-card"
                      style={{ marginBottom: "8px", padding: "12px", cursor: "pointer", borderColor: "var(--border)" }}
                      onClick={() => {
                        setSelectedNodeId(d.id);
                        setView("graph");
                      }}
                    >
                      <div style={{ display: "flex", gap: "6px", alignItems: "flex-start", marginBottom: "4px" }}>
                        <span>{typeIcon[d.group] || "📄"}</span>
                        <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--text)", lineHeight: 1.4 }}>
                          {label}
                        </span>
                      </div>
                      <div style={{ fontSize: "11px", color: "var(--text3)", textAlign: "right" }}>
                        <span>{dateStr}</span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
