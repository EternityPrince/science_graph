"use client";

import { useEffect, useState } from "react";
import { useStore } from "@/lib/store";
import DashboardPanel from "@/components/ui/DashboardPanel";
import GraphCanvas from "@/components/graph/GraphCanvas";
import Sidebar from "@/components/ui/Sidebar";
import ChatInterface from "@/components/ui/ChatInterface";
import NotesPanel from "@/components/ui/NotesPanel";
import ChronologyPanel from "@/components/ui/ChronologyPanel";
import UploadPanel from "@/components/ui/UploadPanel";

const filterGroups = [
  { id: "paper", label: "Статьи" },
  { id: "note", label: "Заметки" },
  { id: "book", label: "Книги" },
  { id: "video", label: "Видео" },
  { id: "webpage", label: "Веб-страницы" },
  { id: "reference", label: "Упомянутые" },
  { id: "author", label: "Авторы" },
  { id: "concept", label: "Концепты" },
  { id: "tag", label: "Теги" },
];

export default function HomePage() {
  const {
    activeView,
    refreshAll,
    graphData,
    filters,
    toggleFilter,
    spacing,
    setSpacing,
    gravity,
    setGravity,
    edgeLength,
    setEdgeLength,
    physicsEnabled,
    setPhysicsEnabled,
    physicsSolver,
    setPhysicsSolver,
    edgeLabels,
    setEdgeLabels,
    setSelectedNodeId,
    maxNodeDegree,
    setMaxNodeDegree,
  } = useStore();

  const [settingsVisible, setSettingsVisible] = useState(false);
  const [graphSearch, setGraphSearch] = useState("");
  const [graphSearchResults, setGraphSearchResults] = useState<any[]>([]);
  const [showGraphSearchResults, setShowGraphSearchResults] = useState(false);

  // Initialize store data on mount
  useEffect(() => {
    refreshAll();
  }, []);

  // Sync click away for settings panel
  useEffect(() => {
    const handleOutsideClick = () => {
      setSettingsVisible(false);
    };
    document.addEventListener("click", handleOutsideClick);
    return () => document.removeEventListener("click", handleOutsideClick);
  }, []);

  // Local graph search debouncing/filter
  useEffect(() => {
    const q = graphSearch.trim().toLowerCase();
    if (!q || !graphData) {
      setGraphSearchResults([]);
      setShowGraphSearchResults(false);
      return;
    }

    const matches = graphData.nodes.filter(n => 
      (n.full_title || n.label || "").toLowerCase().includes(q)
    ).slice(0, 10);

    setGraphSearchResults(matches);
    setShowGraphSearchResults(matches.length > 0);
  }, [graphSearch, graphData]);

  const selectGraphNode = (nodeId: string) => {
    setSelectedNodeId(nodeId);
    setGraphSearch("");
    setShowGraphSearchResults(false);
  };

  const typeIcon: Record<string, string> = {
    paper: "📄",
    note: "📝",
    book: "📚",
    video: "🎥",
    webpage: "🌐",
    author: "👥",
    concept: "🧬",
    tag: "🏷️",
    reference: "📎"
  };

  return (
    <main style={{ display: "flex", flex: 1, overflow: "hidden", position: "relative" }}>
      {activeView === "dashboard" && <DashboardPanel />}
      
      {activeView === "chat" && <ChatInterface />}

      {activeView === "notes" && <NotesPanel />}

      {activeView === "chronology" && <ChronologyPanel />}

      {activeView === "upload" && <UploadPanel />}

      {activeView === "graph" && (
        <div id="view-graph" className="main-view active">
          <div id="graph-wrap">
            {/* Graph Canvas */}
            <GraphCanvas />

            {/* Filter chips */}
            <div className="graph-controls" id="filter-chips">
              {filterGroups.map((group) => (
                <div
                  key={group.id}
                  className={`filter-chip ${filters.has(group.id) ? "active" : "inactive"}`}
                  onClick={() => toggleFilter(group.id)}
                >
                  <span className="chip-dot" style={{ background: `var(--col-${group.id})` }}></span>
                  {group.label}
                </div>
              ))}
            </div>

            {/* Floating Graph Search Box */}
            <div className="graph-search-container">
              <div className="graph-search-input-wrap">
                <span className="graph-search-icon">🔍</span>
                <input
                  id="graph-search-input"
                  type="text"
                  value={graphSearch}
                  onChange={(e) => setGraphSearch(e.target.value)}
                  onFocus={() => setShowGraphSearchResults(graphSearchResults.length > 0)}
                  onBlur={() => setTimeout(() => setShowGraphSearchResults(false), 200)}
                  placeholder="Найти узел на графе..."
                  autoComplete="off"
                />
              </div>
              {showGraphSearchResults && (
                <div className="graph-search-results visible">
                  {graphSearchResults.map((n) => (
                    <div
                      key={n.id}
                      className="graph-search-result-item"
                      onClick={() => selectGraphNode(n.id)}
                    >
                      <span>{typeIcon[n.group] || "📄"}</span>
                      <span>{n.full_title || n.label}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Floating settings toggle button */}
            <button
              className="settings-toggle-btn"
              onClick={(e) => {
                e.stopPropagation();
                setSettingsVisible(!settingsVisible);
              }}
              title="Настройки графа"
            >
              ⚙️
            </button>

            {/* Glassmorphic settings panel */}
            <div
              id="graph-settings"
              className={`graph-settings-panel ${settingsVisible ? "visible" : ""}`}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="settings-header">
                <span>⚙️ Настройки графа</span>
                <button className="btn-close-settings" onClick={() => setSettingsVisible(false)}>
                  ×
                </button>
              </div>
              <div className="settings-body">
                {/* Physics Solver */}
                <div className="settings-group">
                  <label htmlFor="physics-solver-select">Метод раскладки</label>
                  <select
                    id="physics-solver-select"
                    className="settings-select"
                    value={physicsSolver}
                    onChange={(e) => setPhysicsSolver(e.target.value as any)}
                  >
                    <option value="barnesHut">Barnes-Hut (классический)</option>
                    <option value="forceAtlas2Based">ForceAtlas2 (равномерный)</option>
                  </select>
                </div>

                {/* Spacing Slider */}
                <div className="settings-group">
                  <div className="slider-label-row">
                    <label htmlFor="node-spacing-range">Расстояние (сила отталкивания)</label>
                    <span className="slider-value" id="node-spacing-val">
                      {spacing}
                    </span>
                  </div>
                  <input
                    type="range"
                    id="node-spacing-range"
                    min="-30000"
                    max="-5000"
                    step="1000"
                    value={spacing}
                    onChange={(e) => setSpacing(parseInt(e.target.value))}
                  />
                </div>

                {/* Gravity Slider */}
                <div className="settings-group">
                  <div className="slider-label-row">
                    <label htmlFor="gravity-range">Центральная гравитация</label>
                    <span className="slider-value" id="gravity-val">
                      {gravity.toFixed(3)}
                    </span>
                  </div>
                  <input
                    type="range"
                    id="gravity-range"
                    min="0.001"
                    max="0.30"
                    step="0.005"
                    value={gravity}
                    onChange={(e) => setGravity(parseFloat(e.target.value))}
                  />
                </div>

                {/* Edge Length Slider */}
                <div className="settings-group">
                  <div className="slider-label-row">
                    <label htmlFor="edge-length-range">Длина связей</label>
                    <span className="slider-value" id="edge-length-val">
                      {edgeLength}
                    </span>
                  </div>
                  <input
                    type="range"
                    id="edge-length-range"
                    min="100"
                    max="400"
                    step="10"
                    value={edgeLength}
                    onChange={(e) => setEdgeLength(parseInt(e.target.value))}
                  />
                </div>

                {/* Max Node Degree Slider */}
                <div className="settings-group">
                  <div className="slider-label-row">
                    <label htmlFor="max-degree-range">Макс. связей у узла (0 - без лимита)</label>
                    <span className="slider-value">
                      {maxNodeDegree === 0 ? "∞" : maxNodeDegree}
                    </span>
                  </div>
                  <input
                    type="range"
                    id="max-degree-range"
                    min="0"
                    max="200"
                    step="5"
                    value={maxNodeDegree}
                    onChange={(e) => setMaxNodeDegree(parseInt(e.target.value))}
                  />
                </div>

                {/* Toggle options */}
                <div className="settings-checkbox-group">
                  <label className="settings-checkbox-label">
                    <input
                      type="checkbox"
                      id="toggle-physics"
                      checked={physicsEnabled}
                      onChange={(e) => setPhysicsEnabled(e.target.checked)}
                    />
                    <span>Включить физику</span>
                  </label>
                </div>

                <div className="settings-checkbox-group">
                  <label className="settings-checkbox-label">
                    <input
                      type="checkbox"
                      id="toggle-edge-labels"
                      checked={edgeLabels}
                      onChange={(e) => setEdgeLabels(e.target.checked)}
                    />
                    <span>Показывать имена связей</span>
                  </label>
                </div>
              </div>
            </div>
          </div>

          {/* Right Sidebar strictly for details */}
          <Sidebar />
        </div>
      )}
    </main>
  );
}
