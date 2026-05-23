"use client";

import { useEffect, useState } from "react";
import { useStore } from "@/lib/store";
import { openLocalFile } from "@/lib/api";
import { Command } from "cmdk";

export default function LibraryPage() {
  const {
    graphData,
    libraryData,
    libraryPage,
    librarySearch,
    fetchLibraryData,
    setLibrarySearch,
    setSelectedNodeId,
    setView,
    askAbout,
    refreshAll,
  } = useStore();

  const [expandedCardId, setExpandedCardId] = useState<string | null>(null);
  const [filterMenuOpen, setFilterMenuOpen] = useState(false);
  const [cmdkSearch, setCmdkSearch] = useState("");
  const [fileStatus, setFileStatus] = useState<Record<string, string>>({});

  useEffect(() => {
    refreshAll();
  }, []);

  const handleSearchChange = (val: string) => {
    setLibrarySearch(val);
    fetchLibraryData(1, val);
  };

  const handlePageChange = (nextPage: number) => {
    fetchLibraryData(nextPage);
  };

  const selectFilterOption = (value: string) => {
    setLibrarySearch(value);
    fetchLibraryData(1, value);
    setFilterMenuOpen(false);
    setCmdkSearch("");
  };

  const handleOpenLocalFile = async (paperId: string, filePath: string) => {
    setFileStatus((prev) => ({ ...prev, [paperId]: "⏳ Открытие..." }));
    try {
      const res = await openLocalFile(filePath);
      setFileStatus((prev) => ({ ...prev, [paperId]: `✅ ${res.message}` }));
      setTimeout(() => {
        setFileStatus((prev) => {
          const copy = { ...prev };
          delete copy[paperId];
          return copy;
        });
      }, 3000);
    } catch (e: any) {
      setFileStatus((prev) => ({ ...prev, [paperId]: `❌ Ошибка: ${e.message}` }));
      setTimeout(() => {
        setFileStatus((prev) => {
          const copy = { ...prev };
          delete copy[paperId];
          return copy;
        });
      }, 4000);
    }
  };

  // Compile autocomplete choices
  const authors = Array.from(
    new Set(
      graphData?.nodes
        .filter((n) => n.group === "author")
        .map((n) => n.full_title || n.label)
        .filter(Boolean) || []
    )
  );

  const concepts = Array.from(
    new Set(
      graphData?.nodes
        .filter((n) => n.group === "concept")
        .map((n) => n.full_title || n.label)
        .filter(Boolean) || []
    )
  );

  const tags = Array.from(
    new Set(
      graphData?.nodes
        .filter((n) => n.group === "tag")
        .map((n) => n.full_title || n.label)
        .filter(Boolean) || []
    )
  );

  const totalResults = libraryData?.total || 0;
  const pageResults = libraryData?.results || [];
  const totalPages = Math.max(1, Math.ceil(totalResults / 10));

  return (
    <div className="library-container">
      <div className="library-header">
        <h1>Библиотека публикаций</h1>
        <p>Управление и поиск по всем проиндексированным материалам базы знаний</p>
      </div>

      {/* Advanced Filter Row */}
      <div className="library-filter-row">
        <input
          type="text"
          className="library-search-input"
          placeholder="Поиск по названию, содержанию, авторам..."
          value={librarySearch}
          onChange={(e) => handleSearchChange(e.target.value)}
        />

        <div style={{ position: "relative" }}>
          <button 
            className={`btn ${filterMenuOpen ? "btn-primary" : "btn-ghost"}`}
            style={{ height: "100%" }}
            onClick={() => setFilterMenuOpen(!filterMenuOpen)}
          >
            🔍 Автодополнение
          </button>
          
          {filterMenuOpen && (
            <div 
              style={{ position: "absolute", top: "calc(100% + 8px)", right: 0, zIndex: 1000, boxShadow: "var(--shadow)" }}
              onMouseLeave={() => setFilterMenuOpen(false)}
            >
              <Command className="cmdk-root" label="Autocomplete Filter">
                <Command.Input 
                  className="cmdk-input" 
                  placeholder="Фильтр по автору, тегу..." 
                  value={cmdkSearch}
                  onValueChange={setCmdkSearch}
                />
                <Command.List className="cmdk-list">
                  <Command.Empty className="cmdk-item">Ничего не найдено.</Command.Empty>
                  
                  {authors.length > 0 && (
                    <Command.Group heading="Авторы">
                      {authors.map((author) => (
                        <Command.Item 
                          key={author} 
                          value={author}
                          className="cmdk-item"
                          onSelect={() => selectFilterOption(author)}
                        >
                          <span>{author}</span>
                          <span className="cmdk-item-type" style={{ borderColor: "var(--col-author)", color: "var(--col-author)" }}>автор</span>
                        </Command.Item>
                      ))}
                    </Command.Group>
                  )}

                  {concepts.length > 0 && (
                    <Command.Group heading="Концепты">
                      {concepts.map((concept) => (
                        <Command.Item 
                          key={concept} 
                          value={concept}
                          className="cmdk-item"
                          onSelect={() => selectFilterOption(concept)}
                        >
                          <span>{concept}</span>
                          <span className="cmdk-item-type" style={{ borderColor: "var(--col-concept)", color: "var(--col-concept)" }}>концепт</span>
                        </Command.Item>
                      ))}
                    </Command.Group>
                  )}

                  {tags.length > 0 && (
                    <Command.Group heading="Теги">
                      {tags.map((tag) => (
                        <Command.Item 
                          key={tag} 
                          value={tag}
                          className="cmdk-item"
                          onSelect={() => selectFilterOption(tag)}
                        >
                          <span>{tag}</span>
                          <span className="cmdk-item-type" style={{ borderColor: "var(--col-tag)", color: "var(--col-tag)" }}>тег</span>
                        </Command.Item>
                      ))}
                    </Command.Group>
                  )}
                </Command.List>
              </Command>
            </div>
          )}
        </div>

        {librarySearch && (
          <button 
            className="btn btn-danger"
            onClick={() => handleSearchChange("")}
          >
            Сбросить фильтр [X]
          </button>
        )}
      </div>

      <div style={{ fontSize: "11px", color: "var(--text3)", textTransform: "uppercase" }}>
        Всего найдено: {totalResults} публикаций
      </div>

      {/* Grid of Interactive Cards */}
      <div className="library-grid">
        {pageResults.length === 0 ? (
          <div style={{ gridColumn: "span 3", textAlign: "center", padding: "40px", color: "var(--text3)" }}>
            Публикации не найдены
          </div>
        ) : (
          pageResults.map((item) => {
            const isExpanded = expandedCardId === item.id;
            const snippet = item.summary 
              ? (item.summary.length > 180 ? item.summary.substring(0, 180) + "..." : item.summary)
              : (item.abstract ? (item.abstract.length > 180 ? item.abstract.substring(0, 180) + "..." : item.abstract) : "Нет краткого описания.");

            return (
              <div key={item.id} className="library-card">
                <div className="library-card-header">
                  <div className="library-card-title">{item.title}</div>
                  <span className={`details-badge badge-${item.source_type}`}>
                    {item.source_type}
                  </span>
                </div>

                <div className="library-card-meta">
                  {item.authors.length > 0 && <div>✍️ Авторы: {item.authors.join(", ")}</div>}
                  {item.year && <div>📅 Год: {item.year}</div>}
                  {item.doi && <div>📎 DOI: {item.doi}</div>}
                </div>

                <div className="library-card-snippet">{snippet}</div>

                {/* Concepts list */}
                {item.concepts.length > 0 && (
                  <div className="tag-list">
                    {item.concepts.map((concept) => (
                      <span 
                        key={concept} 
                        className="tag"
                        onClick={() => selectFilterOption(concept)}
                      >
                        {concept}
                      </span>
                    ))}
                  </div>
                )}

                {/* Expandable details */}
                {isExpanded && (
                  <div className="library-card-expanded-content">
                    {item.abstract && (
                      <div>
                        <strong>Аннотация:</strong>
                        <p style={{ marginTop: "4px", color: "var(--text2)", lineHeight: "1.6" }}>{item.abstract}</p>
                      </div>
                    )}
                    {item.summary && (
                      <div>
                        <strong>Подробное содержание (LLM Summary):</strong>
                        <p style={{ marginTop: "4px", color: "var(--text2)", lineHeight: "1.6" }}>{item.summary}</p>
                      </div>
                    )}
                    {item.tags.length > 0 && (
                      <div>
                        <strong>Теги:</strong>
                        <div className="tag-list" style={{ marginTop: "4px" }}>
                          {item.tags.map((tag) => (
                            <span 
                              key={tag} 
                              className="tag" 
                              style={{ borderColor: "var(--col-tag)", color: "var(--col-tag)" }}
                              onClick={() => selectFilterOption(tag)}
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Interactive Card Action Footer */}
                <div style={{ display: "flex", gap: "10px", marginTop: "auto", paddingTop: "12px", borderTop: "1px solid var(--border-solid)" }}>
                  <button 
                    className="btn btn-ghost btn-sm"
                    style={{ flex: 1 }}
                    onClick={() => setExpandedCardId(isExpanded ? null : item.id)}
                  >
                    {isExpanded ? "Свернуть" : "Развернуть"}
                  </button>

                  {item.file_path && (
                    <button 
                      className="btn btn-ghost btn-sm"
                      style={{ flex: 1 }}
                      onClick={() => handleOpenLocalFile(item.id, item.file_path!)}
                    >
                      {fileStatus[item.id] || "Открыть файл"}
                    </button>
                  )}

                  <button 
                    className="btn btn-primary btn-sm"
                    onClick={() => askAbout(item.title)}
                  >
                    💬 RAG
                  </button>
                </div>
              </div>
            );
          })
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
          <span style={{ fontSize: "12px", fontWeight: "bold" }}>
            Страница {libraryPage} из {totalPages}
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
    </div>
  );
}
