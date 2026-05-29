"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useStore } from "@/lib/store";
import { fetchPaperDetails, openLocalFile } from "@/lib/api";
import { PaperDetailResponse, LibraryPaperItem } from "@/lib/types";
import { motion, AnimatePresence } from "framer-motion";
import WikiLinkParser from "./WikiLinkParser";


interface WorkCardProps {
  item: LibraryPaperItem;
  isExpanded: boolean;
  onToggleExpand: () => void;
  onOpenDetails: () => void;
}

export default function WorkCard({ item, isExpanded, onToggleExpand, onOpenDetails }: WorkCardProps) {
  const router = useRouter();
  const {
    addLibraryFilter,
    askAbout,
    setSelectedNodeId,
    setView,
    setActiveDocument,
  } = useStore();


  const [details, setDetails] = useState<PaperDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [fileStatus, setFileStatus] = useState<string | null>(null);

  // Load deep metadata dynamically when expanded
  useEffect(() => {
    if (isExpanded && !details) {
      setLoading(true);
      fetchPaperDetails(item.id)
        .then((res) => {
          if (res.type === "paper") {
            setDetails(res);
          }
        })
        .catch((err) => {
          console.error("Error loading paper details inside card:", err);
        })
        .finally(() => {
          setLoading(false);
        });
    }
  }, [isExpanded, item.id, details]);

  const handleOpenLocalFile = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!item.file_path) return;
    setFileStatus("⏳...");
    try {
      const res = await openLocalFile(item.file_path);
      setFileStatus(`✅`);
      setTimeout(() => setFileStatus(null), 3000);
    } catch (err: any) {
      setFileStatus(`❌`);
      setTimeout(() => setFileStatus(null), 4000);
    }
  };

  const handleAuthorClick = (e: React.MouseEvent, author: string) => {
    e.stopPropagation();
    addLibraryFilter({ type: "author", value: author });
  };

  const handleConceptClick = (e: React.MouseEvent, concept: string) => {
    e.stopPropagation();
    addLibraryFilter({ type: "concept", value: concept });
  };

  const handleTagClick = (e: React.MouseEvent, tag: string) => {
    e.stopPropagation();
    addLibraryFilter({ type: "tag", value: tag });
  };

  const typeIcon: Record<string, string> = {
    paper: "📄",
    note: "📝",
    book: "📚",
    video: "🎥",
    webpage: "🌐",
  };

  const typeLabel: Record<string, string> = {
    paper: "Paper",
    note: "Note",
    book: "Book",
    video: "Video",
    webpage: "Web",
  };

  // Compile summary snippet (300-500 chars)
  const summarySnippet = (() => {
    const txt = item.summary || item.abstract || "Нет краткого описания.";
    if (txt.length > 380) {
      return txt.substring(0, 380) + "...";
    }
    return txt;
  })();

  return (
    <div
      className={`library-card ${isExpanded ? "expanded" : ""}`}
      style={{
        height: "100%",
        fontFamily: "'JetBrains Mono', monospace",
        boxSizing: "border-box",
      }}
    >
      {/* Visual lab indicator block in top left */}
      <div
        style={{
          position: "absolute",
          top: "-2px",
          left: "-2px",
          width: "8px",
          height: "8px",
          background: isExpanded ? "var(--accent)" : "var(--border)",
        }}
      />

      {/* Top Row: Icon, Title, Year */}
      <div className="library-card-header" onClick={onToggleExpand} style={{ cursor: "pointer" }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: "10px", flex: 1 }}>
          <span style={{ fontSize: "18px" }} title={typeLabel[item.source_type]}>
            {typeIcon[item.source_type] || "📄"}
          </span>
          <div className="library-card-title">{item.title}</div>
        </div>
        {item.year && (
          <span
            style={{
              fontSize: "11px",
              fontFamily: "'JetBrains Mono', monospace",
              color: "var(--accent)",
              border: "1px solid var(--accent)",
              padding: "2px 6px",
              marginLeft: "12px",
              flexShrink: 0,
            }}
          >
            {item.year}
          </span>
        )}
      </div>

      {/* Sub-Header: Clickable Authors */}
      <div
        style={{
          fontSize: "11px",
          color: "var(--text3)",
          marginTop: "8px",
          display: "flex",
          flexWrap: "wrap",
          gap: "4px",
          alignItems: "center",
        }}
      >
        <span>✍️:</span>
        {item.authors.length > 0 ? (
          item.authors.map((author, index) => (
            <React.Fragment key={author}>
              <span
                onClick={(e) => handleAuthorClick(e, author)}
                style={{
                  color: "var(--text2)",
                  cursor: "pointer",
                  textDecoration: "underline",
                }}
                className="hover-accent"
              >
                {author}
              </span>
              {index < item.authors.length - 1 && <span style={{ color: "var(--text3)" }}>,</span>}
            </React.Fragment>
          ))
        ) : (
          <span style={{ fontStyle: "italic" }}>Неизвестно</span>
        )}
      </div>

      {/* Tags: Top 3 concepts/tags */}
      <div className="tag-list" style={{ marginTop: "12px", gap: "6px" }}>
        {item.concepts.slice(0, 2).map((concept) => (
          <span
            key={concept}
            className="tag tag-concept"
            onClick={(e) => handleConceptClick(e, concept)}
          >
            {concept}
          </span>
        ))}
        {item.tags.slice(0, 1).map((tag) => (
          <span
            key={tag}
            className="tag tag-tag"
            onClick={(e) => handleTagClick(e, tag)}
          >
            {tag}
          </span>
        ))}
      </div>

      {/* Expandable Panel */}
      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            style={{ overflow: "hidden" }}
          >
            <div className="library-card-expanded-content" style={{ marginTop: "16px" }}>
              {/* AI Summary Snippet */}
              <div style={{ marginBottom: "16px" }}>
                <strong style={{ fontSize: "11px", color: "var(--accent)" }}>🧬 RESUME:</strong>
                <div style={{ marginTop: "4px", fontSize: "12px", color: "var(--text2)", lineHeight: "1.5" }}>
                  <WikiLinkParser text={summarySnippet} />
                </div>

                <button
                  className="btn btn-ghost btn-sm"
                  style={{ marginTop: "8px", fontSize: "10px", padding: "2px 6px" }}
                  onClick={(e) => {
                    e.stopPropagation();
                    onOpenDetails();
                  }}
                >
                  Читать подробнее »
                </button>
              </div>

              {/* Deep Metadata */}
              {loading ? (
                <div style={{ color: "var(--text3)", fontSize: "11px" }}>
                  ⏳ Загрузка связей и DOI...
                </div>
              ) : (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "8px",
                    borderTop: "1px solid #222632",
                    paddingTop: "12px",
                    fontSize: "11px",
                  }}
                >
                  {details?.doi && (
                    <div>
                      <span style={{ color: "var(--text3)" }}>DOI:</span>{" "}
                      <a
                        href={`https://doi.org/${details.doi}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ color: "var(--accent)", textDecoration: "underline" }}
                        onClick={(e) => e.stopPropagation()}
                      >
                        {details.doi}
                      </a>
                    </div>
                  )}

                  {details?.created_at && (
                    <div>
                      <span style={{ color: "var(--text3)" }}>ADDED:</span>{" "}
                      <span>{new Date(details.created_at).toLocaleDateString("ru-RU")}</span>
                    </div>
                  )}

                  {/* Connectivity Counters */}
                  <div style={{ display: "flex", gap: "16px", color: "var(--text2)" }}>
                    <div>
                      <span style={{ color: "var(--accent)" }}>●</span> Cited By:{" "}
                      <strong>{details?.cited_by?.length || 0}</strong>
                    </div>
                    <div>
                      <span style={{ color: "var(--accent2)" }}>●</span> References:{" "}
                      <strong>{details?.citations?.length || 0}</strong>
                    </div>
                  </div>

                  {/* Full list of concepts (up to 7-10) */}
                  {details?.concepts && details.concepts.length > 0 && (
                    <div style={{ marginTop: "4px" }}>
                      <span style={{ color: "var(--text3)", display: "block", marginBottom: "4px" }}>
                        CONCEPTS:
                      </span>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                        {details.concepts.slice(0, 8).map((c) => (
                          <span
                            key={c.id}
                            className="tag tag-concept"
                            style={{
                              fontSize: "9px",
                              padding: "1px 4px",
                            }}
                            onClick={(e) => handleConceptClick(e, c.name)}
                          >
                            {c.name}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Footer controls */}
      <div
        style={{
          display: "flex",
          gap: "8px",
          marginTop: "auto",
          paddingTop: "14px",
          borderTop: "1px solid #222632",
        }}
      >
        <button
          className="btn btn-ghost btn-sm"
          style={{ flex: 1, fontSize: "11px", textTransform: "uppercase" }}
          onClick={(e) => {
            e.stopPropagation();
            onToggleExpand();
          }}
        >
          {isExpanded ? "Свернуть" : "Развернуть"}
        </button>

        {item.file_path && (
          <button
            className="btn btn-ghost btn-sm"
            style={{ minWidth: "36px", padding: 0, fontSize: "11px" }}
            onClick={handleOpenLocalFile}
            title="Открыть локальный файл"
          >
            {fileStatus || "📂"}
          </button>
        )}

        <button
          className="btn btn-ghost btn-sm"
          style={{ fontSize: "11px", textTransform: "uppercase" }}
          onClick={(e) => {
            e.stopPropagation();
            setSelectedNodeId(item.id);
            setView("graph");
            router.push("/");
          }}
        >
          🗺️ Graph
        </button>

        <button
          className="btn btn-primary btn-sm"
          style={{ fontSize: "11px", textTransform: "uppercase" }}
          onClick={(e) => {
            e.stopPropagation();
            setActiveDocument(item.id, item.title);
            const store = useStore.getState();
            store.setChatInput(`Проанализируй эту статью и выдели основные тезисы.`);
            setView("chat");
            router.push("/");
          }}
        >
          💬 RAG
        </button>
      </div>
    </div>
  );
}
