"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useStore } from "@/lib/store";
import { fetchPaperDetails, openLocalFile } from "@/lib/api";
import { PaperDetailResponse } from "@/lib/types";
import WikiLinkParser from "./WikiLinkParser";
import { motion, AnimatePresence } from "framer-motion";

interface DetailSheetProps {
  paperId: string | null;
  onClose: () => void;
  onOpenReader?: (id: string) => void;
}

export default function DetailSheet({ paperId, onClose, onOpenReader }: DetailSheetProps) {
  const router = useRouter();
  const { setView, setSelectedNodeId } = useStore();
  const [details, setDetails] = useState<PaperDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [fileStatus, setFileStatus] = useState<string | null>(null);

  useEffect(() => {
    if (!paperId) {
      setDetails(null);
      return;
    }

    setLoading(true);
    setFileStatus(null);
    fetchPaperDetails(paperId)
      .then((res) => {
        if (res.type === "paper") {
          setDetails(res);
        } else {
          setDetails(null);
        }
      })
      .catch((err) => {
        console.error("Error loading paper details:", err);
        setDetails(null);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [paperId]);

  const handleOpenLocalFile = async () => {
    if (!details?.file_path) return;
    setFileStatus("⏳ Открытие...");
    try {
      const res = await openLocalFile(details.file_path);
      setFileStatus(`✅ ${res.message}`);
      setTimeout(() => setFileStatus(null), 3000);
    } catch (e: any) {
      setFileStatus(`❌ Ошибка: ${e.message}`);
      setTimeout(() => setFileStatus(null), 4000);
    }
  };

  const handleFocusOnGraph = () => {
    if (!details) return;
    setSelectedNodeId(details.id);
    setView("graph");
    router.push("/");
  };

  const handleAskAI = () => {
    if (!details) return;
    const store = useStore.getState();
    store.setActiveDocument(details.id, details.title);
    store.setChatInput(`Проанализируй эту статью и выдели основные тезисы.`);
    store.setView("chat");
    router.push("/");
  };

  return (
    <AnimatePresence>
      {paperId && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: "rgba(0, 0, 0, 0.6)",
              backdropFilter: "blur(4px)",
              zIndex: 900,
            }}
          />

          {/* Sliding Sheet */}
          <motion.div
            className="library-detail-sheet"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "tween", duration: 0.25, ease: "easeOut" }}
            style={{
              position: "fixed",
              top: 0,
              right: 0,
              height: "100%",
              width: "min(640px, 100vw)",
              backgroundColor: "var(--surface)",
              borderLeft: "2px solid var(--accent)",
              zIndex: 1000,
              boxShadow: "-8px 0px 0px rgba(7, 8, 11, 0.5)",
              display: "flex",
              flexDirection: "column",
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            {/* Header */}
            <div
              style={{
                padding: "24px",
                borderBottom: "var(--border-solid)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                backgroundColor: "var(--surface2)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontSize: "18px" }}>🧪</span>
                <span
                  style={{
                    fontSize: "11px",
                    fontWeight: "bold",
                    color: "var(--accent)",
                    textTransform: "uppercase",
                    letterSpacing: "1px",
                  }}
                >
                  Карточка Документа // METADATA
                </span>
              </div>
              <button
                onClick={onClose}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "var(--text)",
                  fontSize: "24px",
                  cursor: "pointer",
                  lineHeight: 1,
                  padding: "4px 8px",
                }}
                className="hover-accent"
              >
                ×
              </button>
            </div>

            {/* Content Area */}
            <div
              style={{
                padding: "24px",
                flex: 1,
                overflowY: "auto",
                display: "flex",
                flexDirection: "column",
                gap: "24px",
              }}
            >
              {loading ? (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    height: "100%",
                    color: "var(--accent)",
                    gap: "12px",
                  }}
                >
                  <span style={{ fontSize: "32px", animation: "spin 2s linear infinite" }}>⚙️</span>
                  <span>ЗАГРУЗКА ДАННЫХ ИЗ БАЗЫ ЗНАНИЙ...</span>
                </div>
              ) : details ? (
                <>
                  {/* Title & Type */}
                  <div>
                    <span
                      className={`details-badge badge-${details.source_type}`}
                      style={{
                        padding: "2px 8px",
                        fontSize: "10px",
                        fontWeight: "bold",
                        textTransform: "uppercase",
                        display: "inline-block",
                        marginBottom: "12px",
                        border: "1px solid currentColor",
                      }}
                    >
                      {details.source_type}
                    </span>
                    <h2
                      style={{
                        fontFamily: "'Lora', serif",
                        fontSize: "22px",
                        fontWeight: 700,
                        color: "var(--text)",
                        lineHeight: 1.35,
                      }}
                    >
                      {details.title}
                    </h2>
                  </div>

                  {/* Authors & Year */}
                  <div
                    style={{
                      padding: "12px",
                      backgroundColor: "var(--surface2)",
                      borderLeft: "2px solid var(--accent)",
                      display: "flex",
                      flexDirection: "column",
                      gap: "6px",
                      fontSize: "12px",
                    }}
                  >
                    <div>
                      <span style={{ color: "var(--text3)" }}>АВТОРЫ:</span>{" "}
                      <span style={{ color: "var(--text)" }}>
                        {details.authors?.join(", ") || "Неизвестно"}
                      </span>
                    </div>
                    {details.year && (
                      <div>
                        <span style={{ color: "var(--text3)" }}>ГОД:</span>{" "}
                        <span style={{ color: "var(--text)" }}>{details.year}</span>
                      </div>
                    )}
                    {details.doi && (
                      <div>
                        <span style={{ color: "var(--text3)" }}>DOI:</span>{" "}
                        <a
                          href={`https://doi.org/${details.doi}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{ color: "var(--accent)", textDecoration: "underline" }}
                        >
                          {details.doi}
                        </a>
                      </div>
                    )}
                    {details.created_at && (
                      <div>
                        <span style={{ color: "var(--text3)" }}>ДОБАВЛЕНО:</span>{" "}
                        <span style={{ color: "var(--text2)" }}>
                          {new Date(details.created_at).toLocaleString("ru-RU")}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* LLM Summary */}
                  {details.summary && (
                    <div>
                      <h3
                        style={{
                          fontSize: "12px",
                          fontWeight: "bold",
                          textTransform: "uppercase",
                          color: "var(--accent)",
                          marginBottom: "8px",
                          borderBottom: "1px solid var(--border)",
                          paddingBottom: "4px",
                        }}
                      >
                        🧬 AI Резюме (LLM Summary)
                      </h3>
                      <WikiLinkParser text={details.summary} />
                    </div>
                  )}

                  {/* Abstract */}
                  {details.abstract && (
                    <div>
                      <h3
                        style={{
                          fontSize: "12px",
                          fontWeight: "bold",
                          textTransform: "uppercase",
                          color: "var(--accent2)",
                          marginBottom: "8px",
                          borderBottom: "1px solid var(--border)",
                          paddingBottom: "4px",
                        }}
                      >
                        📄 Аннотация (Abstract)
                      </h3>
                      <WikiLinkParser text={details.abstract} />
                    </div>
                  )}

                  {/* Full list of concepts */}
                  {details.concepts && details.concepts.length > 0 && (
                    <div>
                      <h3
                        style={{
                          fontSize: "12px",
                          fontWeight: "bold",
                          textTransform: "uppercase",
                          color: "var(--text)",
                          marginBottom: "12px",
                          borderBottom: "1px solid var(--border)",
                          paddingBottom: "4px",
                        }}
                      >
                        🧬 Связанные Концепты ({details.concepts.length})
                      </h3>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                        {details.concepts.map((c) => (
                          <span
                            key={c.id}
                            className="tag"
                            style={{
                              borderColor: "var(--col-concept)",
                              color: "var(--col-concept)",
                              cursor: "default",
                            }}
                          >
                            {c.name}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Full list of tags */}
                  {details.tags && details.tags.length > 0 && (
                    <div>
                      <h3
                        style={{
                          fontSize: "12px",
                          fontWeight: "bold",
                          textTransform: "uppercase",
                          color: "var(--text)",
                          marginBottom: "12px",
                          borderBottom: "1px solid var(--border)",
                          paddingBottom: "4px",
                        }}
                      >
                        🏷️ Теги ({details.tags.length})
                      </h3>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                        {details.tags.map((t) => (
                          <span
                            key={t.id}
                            className="tag"
                            style={{
                              borderColor: "var(--col-tag)",
                              color: "var(--col-tag)",
                              cursor: "default",
                            }}
                          >
                            {t.name}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Connectivity Counters */}
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: "16px",
                      marginTop: "12px",
                    }}
                  >
                    <div
                      style={{
                        border: "var(--border-solid)",
                        padding: "12px",
                        textAlign: "center",
                      }}
                    >
                      <div style={{ fontSize: "20px", fontWeight: "bold", color: "var(--accent)" }}>
                        {details.cited_by?.length || 0}
                      </div>
                      <div style={{ fontSize: "10px", color: "var(--text3)", textTransform: "uppercase" }}>
                        Цитируется по базе (Cited By)
                      </div>
                    </div>
                    <div
                      style={{
                        border: "var(--border-solid)",
                        padding: "12px",
                        textAlign: "center",
                      }}
                    >
                      <div style={{ fontSize: "20px", fontWeight: "bold", color: "var(--accent2)" }}>
                        {details.citations?.length || 0}
                      </div>
                      <div style={{ fontSize: "10px", color: "var(--text3)", textTransform: "uppercase" }}>
                        Источники (References)
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div style={{ padding: "40px 0", textAlign: "center", color: "var(--text3)" }}>
                  НЕ УДАЛОСЬ ЗАГРУЗИТЬ ИНФОРМАЦИЮ О ДОКУМЕНТЕ
                </div>
              )}
            </div>

            {/* Action Footer */}
            {details && (
              <div
                style={{
                  padding: "24px",
                  borderTop: "var(--border-solid)",
                  backgroundColor: "var(--surface2)",
                  display: "flex",
                  flexDirection: "column",
                  gap: "12px",
                }}
              >
                <div style={{ display: "flex", gap: "12px" }}>
                  <button
                    className="btn btn-primary"
                    style={{ flex: 1, textTransform: "uppercase" }}
                    onClick={handleFocusOnGraph}
                  >
                    🗺️ Фокус на Графе
                  </button>
                  <button
                    className="btn btn-ghost"
                    style={{ flex: 1, textTransform: "uppercase" }}
                    onClick={handleAskAI}
                  >
                    💬 Спросить AI (RAG)
                  </button>
                </div>
                {onOpenReader && !details.properties?.is_placeholder && (
                  <button
                    className="btn btn-primary"
                    style={{
                      width: "100%",
                      backgroundColor: "var(--accent)",
                      color: "#fff",
                      textTransform: "uppercase",
                      display: "flex",
                      justifyContent: "center",
                      alignItems: "center",
                      gap: "8px",
                      fontWeight: "bold",
                    }}
                    onClick={() => onOpenReader(details.id)}
                  >
                    📖 Режим чтения (Reader Mode)
                  </button>
                )}
                {details.file_path && (
                  <button
                    className="btn btn-ghost"
                    style={{
                      width: "100%",
                      borderColor: "var(--border)",
                      color: "var(--text2)",
                      textTransform: "uppercase",
                      display: "flex",
                      justifyContent: "center",
                      alignItems: "center",
                      gap: "8px",
                    }}
                    onClick={handleOpenLocalFile}
                  >
                    📂 {fileStatus || "Открыть Локальный Файл"}
                  </button>
                )}
              </div>
            )}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
