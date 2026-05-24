"use client";

import React, { useState, useMemo } from "react";
import { useStore } from "@/lib/store";
import { Command } from "cmdk";

export default function FilterBar() {
  const {
    graphData,
    librarySearch,
    setLibrarySearch,
    libraryFilters,
    addLibraryFilter,
    removeLibraryFilter,
    librarySourceTypes,
    toggleLibrarySourceType,
    clearLibraryFilters,
    fromDate,
    setFromDate,
    toDate,
    setToDate,
    heatmapDate,
    setHeatmapDate,
    fetchLibraryData,
  } = useStore();

  const [cmdkSearch, setCmdkSearch] = useState("");
  const [cmdkOpen, setCmdkOpen] = useState(false);

  // Compile autocomplete suggestions from graph nodes
  const graphNodes = useMemo(() => graphData?.nodes || [], [graphData]);

  const authorsList = useMemo(() => {
    return Array.from(
      new Set(
        graphNodes
          .filter((n) => n.group === "author")
          .map((n) => n.full_title || n.label)
          .filter(Boolean)
      )
    ).sort();
  }, [graphNodes]);

  const conceptsList = useMemo(() => {
    return Array.from(
      new Set(
        graphNodes
          .filter((n) => n.group === "concept")
          .map((n) => n.full_title || n.label)
          .filter(Boolean)
      )
    ).sort();
  }, [graphNodes]);

  const tagsList = useMemo(() => {
    return Array.from(
      new Set(
        graphNodes
          .filter((n) => n.group === "tag")
          .map((n) => n.full_title || n.label)
          .filter(Boolean)
      )
    ).sort();
  }, [graphNodes]);

  const handleSelectAutocomplete = (type: "author" | "concept" | "tag", value: string) => {
    addLibraryFilter({ type, value });
    setCmdkSearch("");
    setCmdkOpen(false);
  };

  const handleTextSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setLibrarySearch(val);
    fetchLibraryData(1, val);
  };

  const handleDateReset = () => {
    setFromDate("");
    setToDate("");
    setHeatmapDate(null);
    fetchLibraryData(1);
  };

  const sourceTypesList = [
    { id: "paper", label: "📄 Paper" },
    { id: "book", label: "📚 Book" },
    { id: "note", label: "📝 Note" },
    { id: "video", label: "🎥 Video" },
    { id: "webpage", label: "🌐 Web" },
  ];

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "16px",
        padding: "20px",
        border: "2px solid #222632",
        backgroundColor: "var(--surface)",
        fontFamily: "'JetBrains Mono', monospace",
      }}
    >
      {/* Search and Autocomplete Input Row */}
      <div style={{ display: "flex", gap: "12px", alignItems: "stretch", flexWrap: "wrap" }}>
        <div style={{ flex: 1, minWidth: "280px", position: "relative", display: "flex" }}>
          <input
            type="text"
            className="library-search-input"
            placeholder="[🔎] Текстовый поиск по названию, авторам, содержанию..."
            value={librarySearch}
            onChange={handleTextSearchChange}
            style={{ width: "100%" }}
          />
        </div>

        {/* Cmdk Command Palette Selector */}
        <div style={{ position: "relative" }}>
          <button
            className={`btn ${cmdkOpen ? "btn-primary" : "btn-ghost"}`}
            style={{ height: "100%", textTransform: "uppercase", fontSize: "12px" }}
            onClick={() => setCmdkOpen(!cmdkOpen)}
          >
            🧬 Выбрать Фильтр [↓]
          </button>

          {cmdkOpen && (
            <div
              style={{
                position: "absolute",
                top: "calc(100% + 8px)",
                right: 0,
                zIndex: 110,
                boxShadow: "var(--shadow)",
              }}
              onMouseLeave={() => setCmdkOpen(false)}
            >
              <Command className="cmdk-root" label="Command Center Autocomplete">
                <Command.Input
                  className="cmdk-input"
                  placeholder="Введите имя автора, концепт или тег..."
                  value={cmdkSearch}
                  onValueChange={setCmdkSearch}
                  autoFocus
                />
                <Command.List className="cmdk-list" style={{ minWidth: "300px" }}>
                  <Command.Empty className="cmdk-item">Совпадений не найдено</Command.Empty>

                  {authorsList.length > 0 && (
                    <Command.Group heading="Авторы">
                      {authorsList
                        .filter((a) => a.toLowerCase().includes(cmdkSearch.toLowerCase()))
                        .slice(0, 5)
                        .map((author) => (
                          <Command.Item
                            key={author}
                            value={author}
                            className="cmdk-item"
                            onSelect={() => handleSelectAutocomplete("author", author)}
                          >
                            <span>{author}</span>
                            <span
                              className="cmdk-item-type"
                              style={{ borderColor: "var(--col-author)", color: "var(--col-author)" }}
                            >
                              автор
                            </span>
                          </Command.Item>
                        ))}
                    </Command.Group>
                  )}

                  {conceptsList.length > 0 && (
                    <Command.Group heading="Концепты">
                      {conceptsList
                        .filter((c) => c.toLowerCase().includes(cmdkSearch.toLowerCase()))
                        .slice(0, 5)
                        .map((concept) => (
                          <Command.Item
                            key={concept}
                            value={concept}
                            className="cmdk-item"
                            onSelect={() => handleSelectAutocomplete("concept", concept)}
                          >
                            <span>{concept}</span>
                            <span
                              className="cmdk-item-type"
                              style={{ borderColor: "var(--col-concept)", color: "var(--col-concept)" }}
                            >
                              концепт
                            </span>
                          </Command.Item>
                        ))}
                    </Command.Group>
                  )}

                  {tagsList.length > 0 && (
                    <Command.Group heading="Теги">
                      {tagsList
                        .filter((t) => t.toLowerCase().includes(cmdkSearch.toLowerCase()))
                        .slice(0, 5)
                        .map((tag) => (
                          <Command.Item
                            key={tag}
                            value={tag}
                            className="cmdk-item"
                            onSelect={() => handleSelectAutocomplete("tag", tag)}
                          >
                            <span>{tag}</span>
                            <span
                              className="cmdk-item-type"
                              style={{ borderColor: "var(--col-tag)", color: "var(--col-tag)" }}
                            >
                              тег
                            </span>
                          </Command.Item>
                        ))}
                    </Command.Group>
                  )}
                </Command.List>
              </Command>
            </div>
          )}
        </div>

        {/* Clear Filters Button */}
        {(librarySearch || libraryFilters.length > 0 || fromDate || toDate || heatmapDate) && (
          <button
            className="btn btn-danger"
            onClick={clearLibraryFilters}
            style={{ textTransform: "uppercase", fontSize: "12px" }}
          >
            Сбросить фильтры [X]
          </button>
        )}
      </div>

      {/* Selected Filters Chips Row */}
      {libraryFilters.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center" }}>
          <span style={{ fontSize: "11px", color: "var(--text3)", textTransform: "uppercase" }}>
            Активные фильтры:
          </span>
          {libraryFilters.map((filter) => {
            const colorClass =
              filter.type === "author"
                ? "var(--col-author)"
                : filter.type === "concept"
                ? "var(--col-concept)"
                : "var(--col-tag)";

            return (
              <div
                key={`${filter.type}-${filter.value}`}
                style={{
                  border: `1px solid ${colorClass}`,
                  color: colorClass,
                  padding: "4px 8px",
                  fontSize: "11px",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  backgroundColor: "rgba(0,0,0,0.2)",
                }}
              >
                <span>
                  {filter.type.toUpperCase()}: {filter.value}
                </span>
                <span
                  style={{ cursor: "pointer", fontWeight: "bold" }}
                  onClick={() => removeLibraryFilter(filter)}
                  className="hover-accent"
                >
                  [×]
                </span>
              </div>
            );
          })}
        </div>
      )}

      {/* Quick Toggles Row (Source Type & Date Range) */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "16px",
          borderTop: "1px dashed #222632",
          paddingTop: "16px",
        }}
      >
        {/* Source Type Toggle Badges */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center" }}>
          <span style={{ fontSize: "11px", color: "var(--text3)", textTransform: "uppercase" }}>
            Тип источника:
          </span>
          {sourceTypesList.map((type) => {
            const isActive = librarySourceTypes.has(type.id);
            return (
              <button
                key={type.id}
                onClick={() => toggleLibrarySourceType(type.id)}
                style={{
                  padding: "4px 10px",
                  fontSize: "11px",
                  border: "1px solid",
                  borderColor: isActive ? `var(--col-${type.id})` : "#222632",
                  color: isActive ? `var(--col-${type.id})` : "var(--text3)",
                  backgroundColor: isActive ? "rgba(0,0,0,0.15)" : "transparent",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}
                className="hover-accent"
              >
                {type.label}
              </button>
            );
          })}
        </div>

        {/* Date Range Logic */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
          <span style={{ fontSize: "11px", color: "var(--text3)", textTransform: "uppercase" }}>
            Интервал дат:
          </span>

          <input
            type="date"
            value={heatmapDate || fromDate}
            onChange={(e) => {
              if (heatmapDate) setHeatmapDate(null);
              setFromDate(e.target.value);
              fetchLibraryData(1);
            }}
            style={{
              backgroundColor: "var(--bg)",
              border: "1px solid #222632",
              color: "var(--text2)",
              fontSize: "11px",
              padding: "2px 6px",
              fontFamily: "'JetBrains Mono', monospace",
              outline: "none",
            }}
            disabled={!!heatmapDate}
          />
          <span style={{ color: "var(--text3)", fontSize: "11px" }}>—</span>
          <input
            type="date"
            value={heatmapDate || toDate}
            onChange={(e) => {
              if (heatmapDate) setHeatmapDate(null);
              setToDate(e.target.value);
              fetchLibraryData(1);
            }}
            style={{
              backgroundColor: "var(--bg)",
              border: "1px solid #222632",
              color: "var(--text2)",
              fontSize: "11px",
              padding: "2px 6px",
              fontFamily: "'JetBrains Mono', monospace",
              outline: "none",
            }}
            disabled={!!heatmapDate}
          />

          {heatmapDate && (
            <span
              style={{
                fontSize: "10px",
                color: "var(--accent)",
                border: "1px solid var(--accent)",
                padding: "2px 6px",
              }}
              title="Выбран конкретный день в календаре"
            >
              [Календарь: {heatmapDate}]
            </span>
          )}

          {(fromDate || toDate || heatmapDate) && (
            <button
              onClick={handleDateReset}
              style={{
                background: "transparent",
                border: "none",
                color: "var(--red)",
                fontSize: "11px",
                cursor: "pointer",
                padding: "2px 4px",
                textDecoration: "underline",
              }}
            >
              Сбросить дату
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
