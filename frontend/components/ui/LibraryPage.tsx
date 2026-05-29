"use client";

import React, { useEffect, useState, useRef } from "react";
import { useSearchParams } from "next/navigation";
import { useStore } from "@/lib/store";
import FilterBar from "./FilterBar";
import WorkCard from "./WorkCard";
import DetailSheet from "./DetailSheet";
import { List } from "react-window";
import { fetchPaperDetails, fetchPaperText } from "@/lib/api";

import { LibraryResponse, LibraryPaperItem } from "@/lib/types";

interface LibraryPageProps {
  initialData?: LibraryResponse | null;
}

export default function LibraryPage({ initialData }: LibraryPageProps) {
  const {
    libraryData: storeLibraryData,
    libraryPage,
    librarySearch,
    fetchLibraryData,
    refreshAll,
    onlyIndexed,
    setOnlyIndexed,
  } = useStore();

  const libraryData = storeLibraryData || initialData;

  const [expandedCardId, setExpandedCardId] = useState<string | null>(null);
  const [selectedDetailPaperId, setSelectedDetailPaperId] = useState<string | null>(null);
  const [layoutMode, setLayoutMode] = useState<"grid" | "list">("grid");

  const [readerPaperId, setReaderPaperId] = useState<string | null>(null);
  const [readerLoading, setReaderLoading] = useState(false);
  const [readerTitle, setReaderTitle] = useState("");
  const [readerChunks, setReaderChunks] = useState<Array<{ id: string; text_content: string; page_number?: number }>>([]);
  const [readerChatInput, setReaderChatInput] = useState("");
  const [readerMessages, setReaderMessages] = useState<Array<{ role: "user" | "agent"; content: string; isStreaming?: boolean }>>([]);
  const [readerChatStreaming, setReaderChatStreaming] = useState(false);
  const readerChatEndRef = useRef<HTMLDivElement>(null);

  const listRef = useRef<any>(null);
  const searchParams = useSearchParams();
  const paperIdParam = searchParams.get("id");

  // Sync "id" URL query parameter to show the details sheet
  useEffect(() => {
    if (paperIdParam) {
      setSelectedDetailPaperId(paperIdParam);
    }
  }, [paperIdParam]);

  // Initialize data on mount
  useEffect(() => {
    const state = useStore.getState();
    
    // Only set initial library data if it has not been loaded in the store yet
    if (!state.libraryData && initialData) {
      useStore.setState({ libraryData: initialData });
    }

    // Always ensure related telemetry is loaded or refreshed on mount
    Promise.all([
      state.fetchStats(),
      state.fetchModels(),
      state.fetchNotes(),
      state.fetchGraphData()
    ]).catch(console.error);

    // If not using initialData (i.e. direct client navigation), trigger refresh
    if (!initialData) {
      refreshAll();
    }
  }, [initialData, refreshAll]);


  // Whenever card expansion changes, reset virtualization heights if supported
  useEffect(() => {
    if (listRef.current && typeof listRef.current.resetAfterIndex === "function") {
      listRef.current.resetAfterIndex(0);
    }
  }, [expandedCardId]);

  // Load paper text for reader mode
  useEffect(() => {
    if (!readerPaperId) {
      setReaderChunks([]);
      setReaderTitle("");
      setReaderMessages([]);
      return;
    }

    setReaderLoading(true);
    
    // Fetch details to get title
    fetchPaperDetails(readerPaperId)
      .then((det) => {
        if (det.type === "paper") {
          setReaderTitle(det.title);
        }
      })
      .catch(console.error);

    // Fetch chunks
    fetchPaperText(readerPaperId)
      .then((res) => {
        setReaderChunks(res.chunks || []);
      })
      .catch((err) => {
        console.error("Error loading reader text:", err);
      })
      .finally(() => {
        setReaderLoading(false);
      });
  }, [readerPaperId]);

  // Scroll to bottom when reader messages change
  useEffect(() => {
    readerChatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [readerMessages]);

  const handlePresetQuery = (q: string) => {
    setReaderChatInput(q);
  };

  const handleReaderChatSend = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = readerChatInput.trim();
    if (!q || readerChatStreaming || !readerPaperId) return;

    setReaderChatInput("");
    setReaderChatStreaming(true);

    const userMsg = { role: "user" as const, content: q };
    setReaderMessages((prev) => [...prev, userMsg]);

    const agentMsg = { role: "agent" as const, content: "", isStreaming: true };
    setReaderMessages((prev) => [...prev, agentMsg]);

    let accumulated = "";

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          limit: 5,
          cloud: false,
          paper_id: readerPaperId
        })
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No reader");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          try {
            const parsed = JSON.parse(raw);
            if (parsed.token) {
              accumulated += parsed.token;
              setReaderMessages((prev) => {
                const next = [...prev];
                const last = next[next.length - 1];
                if (last && last.role === "agent") {
                  last.content = accumulated;
                }
                return next;
              });
            } else if (parsed.error) {
              accumulated += `\n[Ошибка: ${parsed.error}]`;
              setReaderMessages((prev) => {
                const next = [...prev];
                const last = next[next.length - 1];
                if (last && last.role === "agent") {
                  last.content = accumulated;
                }
                return next;
              });
            }
          } catch (err) {
            // parsing error
          }
        }
      }
    } catch (err: any) {
      console.error("Reader chat error:", err);
      accumulated += `\n[Не удалось получить ответ: ${err.message}]`;
    } finally {
      setReaderChatStreaming(false);
      setReaderMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last && last.role === "agent") {
          last.isStreaming = false;
          last.content = accumulated || "Нет ответа.";
        }
        return next;
      });
    }
  };

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
    const item: LibraryPaperItem = pageResults[index];
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
            className={`btn ${onlyIndexed ? "btn-primary" : "btn-ghost"}`}
            onClick={() => setOnlyIndexed(!onlyIndexed)}
            style={{ fontSize: "11px", textTransform: "uppercase" }}
          >
            {onlyIndexed ? "[🔍 ТОЛЬКО ПРОИНДЕКСИРОВАННЫЕ]" : "[🌐 ВСЕ МАТЕРИАЛЫ И ССЫЛКИ]"}
          </button>
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
            {pageResults.map((item: LibraryPaperItem) => (
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
        onOpenReader={(id) => {
          setSelectedDetailPaperId(null);
          setReaderPaperId(id);
        }}
      />

      {/* Reader Mode Overlay */}
      {readerPaperId && (
        <div style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100vw",
          height: "100vh",
          zIndex: 2000,
          backgroundColor: "#07080b",
          display: "flex",
          flexDirection: "column",
          fontFamily: "'JetBrains Mono', monospace",
          color: "var(--text)"
        }}>
          {/* Top Bar */}
          <div style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "16px 24px",
            borderBottom: "2px solid var(--border)",
            backgroundColor: "var(--surface)"
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <span style={{ fontSize: "20px" }}>📖</span>
              <div>
                <span style={{ fontSize: "10px", color: "var(--accent)", textTransform: "uppercase" }}>Режим чтения (Reader Mode)</span>
                <h2 style={{ fontSize: "16px", margin: 0, fontFamily: "'Lora', serif" }}>{readerTitle || "Загрузка..."}</h2>
              </div>
            </div>
            <button 
              className="btn btn-ghost" 
              onClick={() => setReaderPaperId(null)}
              style={{ textTransform: "uppercase" }}
            >
              [× ЗАКРЫТЬ]
            </button>
          </div>

          {/* Main Grid: Split Screen */}
          <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
            
            {/* Left: Text Content (60% width) */}
            <div style={{
              flex: "0 0 60%",
              overflowY: "auto",
              padding: "40px",
              borderRight: "2px solid var(--border)",
              backgroundColor: "#0b0c10"
            }}>
              {readerLoading ? (
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: "16px", color: "var(--accent)" }}>
                  <span style={{ fontSize: "32px", animation: "spin 2s linear infinite" }}>⚙️</span>
                  <span>ЗАГРУЗКА ТЕКСТА ИЗ БАЗЫ ЗНАНИЙ...</span>
                </div>
              ) : readerChunks.length === 0 ? (
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text3)" }}>
                  [!] В базе знаний нет извлеченного текста для этого документа
                </div>
              ) : (
                <div style={{
                  maxWidth: "720px",
                  margin: "0 auto",
                  fontFamily: "'Lora', Georgia, serif",
                  fontSize: "17px",
                  lineHeight: "1.75",
                  color: "#c9cde0"
                }}>
                  {readerChunks.map((chunk) => (
                    <div key={chunk.id} style={{ marginBottom: "32px", position: "relative" }}>
                      {chunk.page_number && (
                        <div style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "10px",
                          fontSize: "10px",
                          fontFamily: "'JetBrains Mono', monospace",
                          color: "var(--text3)",
                          marginBottom: "16px",
                          textTransform: "uppercase"
                        }}>
                          <div style={{ flex: 1, height: "1px", background: "var(--border)" }} />
                          <span>Страница {chunk.page_number}</span>
                          <div style={{ flex: 1, height: "1px", background: "var(--border)" }} />
                        </div>
                      )}
                      <p style={{ whiteSpace: "pre-wrap" }}>{chunk.text_content}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Right: Focused Chat (40% width) */}
            <div style={{
              flex: "0 0 40%",
              display: "flex",
              flexDirection: "column",
              backgroundColor: "var(--surface)",
              overflow: "hidden"
            }}>
              {/* Chat Header */}
              <div style={{
                padding: "16px",
                borderBottom: "1px solid var(--border)",
                backgroundColor: "var(--surface2)",
                fontSize: "11px",
                textTransform: "uppercase"
              }}>
                💬 Задать вопросы по этой публикации (RAG)
              </div>

              {/* Chat Messages */}
              <div style={{
                flex: 1,
                overflowY: "auto",
                padding: "20px",
                display: "flex",
                flexDirection: "column",
                gap: "16px"
              }}>
                {readerMessages.length === 0 ? (
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text3)", textAlign: "center", fontSize: "12px", padding: "20px" }}>
                    <span>🧬 Спросите AI о содержании этой статьи. Например:</span>
                    <span 
                      onClick={() => handlePresetQuery("Выдели основные тезисы и цели этого исследования")}
                      style={{ color: "var(--accent)", cursor: "pointer", textDecoration: "underline", marginTop: "12px" }}
                    >
                      "Выдели основные тезисы и цели этого исследования"
                    </span>
                    <span 
                      onClick={() => handlePresetQuery("Какая методология используется авторами?")}
                      style={{ color: "var(--accent)", cursor: "pointer", textDecoration: "underline", marginTop: "8px" }}
                    >
                      "Какая методология используется авторами?"
                    </span>
                  </div>
                ) : (
                  readerMessages.map((msg, idx) => (
                    <div key={idx} style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: "4px",
                      alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
                      maxWidth: "90%"
                    }}>
                      <span style={{ fontSize: "9px", color: "var(--text3)", textTransform: "uppercase", alignSelf: msg.role === "user" ? "flex-end" : "flex-start" }}>
                        {msg.role === "user" ? "Вы" : "ИИ-Ассистент"}
                      </span>
                      <div style={{
                        padding: "12px 16px",
                        background: msg.role === "user" ? "rgba(99, 102, 241, 0.15)" : "var(--surface2)",
                        border: msg.role === "user" ? "1px solid var(--accent)" : "1px solid var(--border)",
                        fontSize: "12px",
                        lineHeight: "1.5",
                        whiteSpace: "pre-wrap"
                      }}>
                        {msg.content}
                        {msg.isStreaming && <span className="typing-cursor"></span>}
                      </div>
                    </div>
                  ))
                )}
                <div ref={readerChatEndRef} />
              </div>

              {/* Chat Input */}
              <div style={{
                padding: "16px",
                borderTop: "1px solid var(--border)",
                backgroundColor: "var(--surface2)"
              }}>
                <form onSubmit={handleReaderChatSend} style={{ display: "flex", gap: "10px" }}>
                  <input
                    type="text"
                    className="form-control"
                    placeholder="Задайте вопрос по тексту публикации..."
                    value={readerChatInput}
                    onChange={(e) => setReaderChatInput(e.target.value)}
                    disabled={readerChatStreaming || readerLoading}
                    style={{ flex: 1, fontSize: "12px", height: "36px", margin: 0 }}
                  />
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={readerChatStreaming || !readerChatInput.trim() || readerLoading}
                    style={{ fontSize: "11px", textTransform: "uppercase" }}
                  >
                    Отправить
                  </button>
                </form>
              </div>

            </div>

          </div>
        </div>
      )}
    </div>
  );
}
