"use client";

import { useState, useRef } from "react";
import { PaperDetailResponse } from "@/lib/types";
import { useStore } from "@/lib/store";
import { parseWikiLinks } from "@/utils/wikiLinks";

interface Props {
  details: PaperDetailResponse;
}

export default function VideoDetails({ details }: Props) {
  const [activeTab, setActiveTab] = useState<"overview" | "themes" | "outline" | "transcript">("overview");
  const graphNodes = useStore((state) => state.graphData?.nodes || []);
  const setSelectedNodeId = useStore((state) => state.setSelectedNodeId);
  const setView = useStore((state) => state.setView);

  const props = details.properties || {};
  const videoId = props.video_id || "";
  const uploader = props.uploader || (details.authors && details.authors[0]) || "Неизвестный автор";
  const duration = props.duration ? `${Math.floor(props.duration / 60)} мин ${props.duration % 60} сек` : "";
  const publishDate = props.created_at ? props.created_at.substring(0, 10) : "";

  // Player URL state to support timestamp seeking
  const [playerUrl, setPlayerUrl] = useState(
    videoId ? `https://www.youtube.com/embed/${videoId}?enablejsapi=1&rel=0` : ""
  );

  const seekVideo = (seconds: number) => {
    if (videoId) {
      setPlayerUrl(`https://www.youtube.com/embed/${videoId}?start=${seconds}&autoplay=1&enablejsapi=1&rel=0`);
    }
  };

  // Click interceptor for wiki-links and timestamps inside tabs
  const handleContentClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement;
    
    // Check if wiki-link
    if (target.classList.contains("wiki-link")) {
      e.preventDefault();
      const nodeId = target.getAttribute("data-node-id");
      if (nodeId) {
        setSelectedNodeId(nodeId);
        setView("graph");
      }
    }
    
    // Check if timestamp-btn
    if (target.classList.contains("timestamp-btn")) {
      e.preventDefault();
      const secondsAttr = target.getAttribute("data-seconds");
      if (secondsAttr) {
        seekVideo(parseInt(secondsAttr, 10));
      }
    }
  };

  const formatTextWithTimestamps = (htmlStr: string) => {
    if (!htmlStr) return "";
    const regex = /\[(?:(\d{1,2}):)?(\d{2}):(\d{2})\]/g;
    return htmlStr.replace(regex, (match, hh, mm, ss) => {
      const hours = hh ? parseInt(hh, 10) : 0;
      const minutes = parseInt(mm, 10);
      const seconds = parseInt(ss, 10);
      const totalSeconds = hours * 3600 + minutes * 60 + seconds;
      return `<button class="timestamp-btn" data-seconds="${totalSeconds}">${match}</button>`;
    });
  };

  const processText = (text: string) => {
    if (!text) return "";
    const withWiki = parseWikiLinks(text, graphNodes);
    return formatTextWithTimestamps(withWiki);
  };

  return (
    <div className="details-row">
      <div className="details-card">
        <h3>
          <span className="details-badge badge-video">Видео</span>
          {details.title}
        </h3>
        {videoId && (
          <div className="video-player-container">
            <iframe 
              id={`yt-player-${videoId}`}
              src={playerUrl} 
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
              allowFullScreen>
            </iframe>
          </div>
        )}
        <div className="details-row" style={{ marginTop: "10px" }}>
          {uploader && <div className="details-field"><div className="details-label">Автор</div><div className="details-value">{uploader}</div></div>}
          {publishDate && <div className="details-field"><div className="details-label">Дата</div><div className="details-value">{publishDate}</div></div>}
          {duration && <div className="details-field"><div className="details-label">Длительность</div><div className="details-value">{duration}</div></div>}
          {props.url && <div className="details-field"><div className="details-label">Ссылка</div><div className="details-value"><a href={props.url} target="_blank" rel="noreferrer">Открыть YouTube ↗</a></div></div>}
        </div>
      </div>

      {(props.video_overview || props.video_themes || props.video_outline || props.transcript) ? (
        <div className="details-card" style={{ padding: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <div className="sidebar-tabs">
            <button 
              className={`sidebar-tab-btn ${activeTab === "overview" ? "active" : ""}`}
              onClick={() => setActiveTab("overview")}
            >
              Обзор
            </button>
            <button 
              className={`sidebar-tab-btn ${activeTab === "themes" ? "active" : ""}`}
              onClick={() => setActiveTab("themes")}
            >
              Темы
            </button>
            <button 
              className={`sidebar-tab-btn ${activeTab === "outline" ? "active" : ""}`}
              onClick={() => setActiveTab("outline")}
            >
              Конспект
            </button>
            <button 
              className={`sidebar-tab-btn ${activeTab === "transcript" ? "active" : ""}`}
              onClick={() => setActiveTab("transcript")}
            >
              Транскрипт
            </button>
          </div>
          
          <div 
            className="sidebar-tab-content active"
            style={{ padding: "16px", display: activeTab === "overview" ? "flex" : "none" }}
            onClick={handleContentClick}
            dangerouslySetInnerHTML={{ __html: processText(props.video_overview || "Нет обзора") }}
          />

          <div 
            className="sidebar-tab-content active"
            style={{ padding: "16px", display: activeTab === "themes" ? "flex" : "none" }}
            onClick={handleContentClick}
            dangerouslySetInnerHTML={{ 
              __html: Array.isArray(props.video_themes) 
                ? props.video_themes.map((t: string) => `<div class="video-theme-item">${processText(t)}</div>`).join("") 
                : processText(props.video_themes || "Нет тем") 
            }}
          />

          <div 
            className="sidebar-tab-content active"
            style={{ padding: "16px", display: activeTab === "outline" ? "flex" : "none" }}
            onClick={handleContentClick}
            dangerouslySetInnerHTML={{ 
              __html: Array.isArray(props.video_outline) 
                ? props.video_outline.map((o: string) => `<div class="video-outline-item">${processText(o)}</div>`).join("") 
                : processText(props.video_outline || "Нет конспекта") 
            }}
          />

          <div 
            className="sidebar-tab-content active"
            style={{ padding: "16px", display: activeTab === "transcript" ? "flex" : "none" }}
            onClick={handleContentClick}
          >
            <div 
              className="video-transcript-box"
              dangerouslySetInnerHTML={{ __html: processText(props.transcript || "Транскрипт отсутствует") }}
            />
          </div>
          <div style={{ fontSize: "10px", color: "var(--text3)", padding: "8px 16px", borderTop: "1px solid var(--border-solid)", background: "var(--surface2)" }}>
            🤖 Обзор сгенерирован LLM
          </div>
        </div>
      ) : details.summary && (
        <div className="details-card">
          <h3>🎥 Обзор видео</h3>
          <div 
            style={{ fontSize: "13px", color: "var(--text2)", lineHeight: "1.6", marginTop: "10px", maxHeight: "250px", overflowY: "auto", background: "var(--surface3)", padding: "10px 12px" }}
            onClick={handleContentClick}
            dangerouslySetInnerHTML={{ __html: processText(details.summary) }}
          />
        </div>
      )}

      {details.concepts && details.concepts.length > 0 && (
        <div className="details-card">
          <h3>🧠 Концепты</h3>
          <div className="tag-list" style={{ marginTop: "10px" }}>
            {details.concepts.map((c) => (
              <span key={c.id} className="tag" onClick={() => setSelectedNodeId(c.id)}>
                {c.name}
              </span>
            ))}
          </div>
        </div>
      )}

      {details.tags && details.tags.length > 0 && (
        <div className="details-card">
          <h3>🏷️ Теги</h3>
          <div className="tag-list" style={{ marginTop: "10px" }}>
            {details.tags.map((t) => (
              <span key={t.id} className="tag" style={{ borderColor: "var(--col-tag)", color: "var(--col-tag)" }} onClick={() => setSelectedNodeId(t.id)}>
                {t.name}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
