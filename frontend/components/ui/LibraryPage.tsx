"use client";

import React, { useEffect, useState, useRef } from "react";
import { useStore } from "@/lib/store";
import FilterBar from "./FilterBar";
import WorkCard from "./WorkCard";
import DetailSheet from "./DetailSheet";
import { List } from "react-window";

export default function LibraryPage() {
  const {
    libraryData,
    libraryPage,
    librarySearch,
    fetchLibraryData,
    refreshAll,
  } = useStore();

  const [expandedCardId, setExpandedCardId] = useState<string | null>(null);
  const [selectedDetailPaperId, setSelectedDetailPaperId] = useState<string | null>(null);
  const [layoutMode, setLayoutMode] = useState<"grid" | "list">("grid");

  const listRef = useRef<any>(null);

  // Initialize data on mount
  useEffect(() => {
    refreshAll();
  }, []);

  // Whenever card expansion changes, reset virtualization heights if supported
  useEffect(() => {
    if (listRef.current && typeof listRef.current.resetAfterIndex === "function") {
      listRef.current.resetAfterIndex(0);
    }
  }, [expandedCardId]);

  const totalResults = libraryData?.total || 0;
  const pageResults = libraryData?.results || [];
  
  // With page limit set to 100, page count calculation:
  const itemsPerPage = 100;
  const totalPages = Math.max(1, Math.ceil(totalResults / itemsPerPage));

  const handlePageChange = (nextPage: number) => {
    setExpandedCardId(null);
    fetchLibraryData(nextPage);
  };

  // Virtualized row height calculations
  const getItemSize = (index: number) => {
    const item = pageResults[index];
    if (!item) return 160;
    const isExpanded = expandedCardId === item.id;
    if (isExpanded) {
      // Expanded card has additional summary snippet, concepts list, and action footer
      return 360 + (item.concepts.length > 0 ? 60 : 0);
    }
    return 150;
  };

  // Virtualized Row component
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => {
    const item = pageResults[index];
    if (!item) return null;

    return (
      <div style={{ ...style, paddingBottom: "16px", boxSizing: "border-box" }}>
        <WorkCard
          item={item}
          isExpanded={expandedCardId === item.id}
          onToggleExpand={() => setExpandedCardId(expandedCardId === item.id ? null : item.id)}
          onOpenDetails={() => setSelectedDetailPaperId(item.id)}
        />
      </div>
    );
  };

  return (
    <div className="library-container" style={{ display: "flex", flexDirection: "column", height: "100%", overflowY: "auto", fontFamily: "'JetBrains Mono', monospace" }}>
      {/* Editorial brutality header */}
      <div 
        className="library-header" 
        style={{
          borderBottom: "var(--border-solid)",
          paddingBottom: "16px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-end",
          flexWrap: "wrap",
          gap: "16px"
        }}
      >
        <div>
          <h1 style={{ textTransform: "uppercase", letterSpacing: "-1px" }}>Библиотека публикаций</h1>
          <p style={{ color: "var(--text3)", fontSize: "12px", marginTop: "4px" }}>
            INDEXED_WORKS_DB // TOTAL_RECORDS: {totalResults}
          </p>
        </div>

        {/* Layout and control toggles */}
        <div style={{ display: "flex", gap: "10px" }}>
          <button 
            className={`btn ${layoutMode === "grid" ? "btn-primary" : "btn-ghost"}`}
            onClick={() => setLayoutMode("grid")}
            style={{ fontSize: "11px", textTransform: "uppercase" }}
          >
            [⊞ GRID]
          </button>
          <button 
            className={`btn ${layoutMode === "list" ? "btn-primary" : "btn-ghost"}`}
            onClick={() => setLayoutMode("list")}
            style={{ fontSize: "11px", textTransform: "uppercase" }}
          >
            [☰ LIST_VIRTUALIZED]
          </button>
        </div>
      </div>

      {/* Advanced Command Center filters */}
      <FilterBar />

      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--text3)", textTransform: "uppercase", marginTop: "8px" }}>
        <span>Канал связи: API_GET_DOCUMENTS // ACTIVE</span>
        <span>Найдено в текущей выборке: {pageResults.length}</span>
      </div>

      {/* Document rendering view */}
      <div style={{ flex: 1, minHeight: "350px", marginTop: "12px" }}>
        {pageResults.length === 0 ? (
          <div style={{ border: "2px dashed #222632", textAlign: "center", padding: "80px", color: "var(--text3)" }}>
            [!] ПО ЗАДАННЫМ КРИТЕРИЯМ ПОИСКА ДОКУМЕНТЫ НЕ НАЙДЕНЫ В БАЗЕ ЗНАНИЙ
          </div>
        ) : layoutMode === "list" ? (
          /* List virtualized container using react-window */
          <div style={{ height: "600px", border: "2px solid #222632", padding: "16px", backgroundColor: "rgba(0,0,0,0.1)" }}>
            <List<any>
              listRef={listRef}
              rowCount={pageResults.length}
              rowHeight={getItemSize}
              rowComponent={Row}
              rowProps={{}}
              style={{ width: "100%", height: 560 }}
            />
          </div>
        ) : (
          /* Brutalist responsive grid layout */
          <div className="library-grid">
            {pageResults.map((item) => (
              <WorkCard
                key={item.id}
                item={item}
                isExpanded={expandedCardId === item.id}
                onToggleExpand={() => setExpandedCardId(expandedCardId === item.id ? null : item.id)}
                onOpenDetails={() => setSelectedDetailPaperId(item.id)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div className="library-pagination">
          <button 
            className="library-pagination-btn"
            disabled={libraryPage === 1}
            onClick={() => handlePageChange(libraryPage - 1)}
          >
            ← Назад
          </button>
          <span style={{ fontSize: "12px", fontWeight: "bold", fontFamily: "'JetBrains Mono', monospace" }}>
            СТРАНИЦА {libraryPage} ИЗ {totalPages}
          </span>
          <button 
            className="library-pagination-btn"
            disabled={libraryPage === totalPages}
            onClick={() => handlePageChange(libraryPage + 1)}
          >
            Вперед →
          </button>
        </div>
      )}

      {/* Detail Slide-out Sheet */}
      <DetailSheet 
        paperId={selectedDetailPaperId} 
        onClose={() => setSelectedDetailPaperId(null)} 
      />
    </div>
  );
}
