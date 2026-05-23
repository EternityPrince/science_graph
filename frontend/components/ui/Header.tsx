"use client";

import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";
import { useEffect, useState, useRef } from "react";
import { useStore } from "@/lib/store";
import { searchPapers } from "@/lib/api";

export default function Header() {
  const router = useRouter();
  const pathname = usePathname();

  const {
    activeView,
    setView,
    stats,
    refreshAll,
    setSelectedNodeId
  } = useStore();

  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [showResults, setShowResults] = useState(false);
  const searchTimerRef = useRef<NodeJS.Timeout | null>(null);

  const handleNavClick = (viewName: 'dashboard' | 'graph' | 'chat' | 'notes' | 'chronology' | 'upload' | 'library') => {
    if (viewName === "library") {
      router.push("/library");
    } else {
      setView(viewName);
      if (pathname !== "/") {
        router.push("/");
      }
    }
  };

  // Debounced search query handler
  useEffect(() => {
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);

    const q = searchQuery.trim();
    if (!q) {
      setSearchResults([]);
      setShowResults(false);
      return;
    }

    searchTimerRef.current = setTimeout(async () => {
      try {
        const res = await searchPapers(q);
        setSearchResults(res.results || []);
        setShowResults(res.results && res.results.length > 0);
      } catch (err) {
        console.error("Search failed:", err);
      }
    }, 300);

    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    };
  }, [searchQuery]);

  const selectSearchResult = (nodeId: string) => {
    setSelectedNodeId(nodeId);
    setShowResults(false);
    setSearchQuery("");
    setView("graph");
    if (pathname !== "/") {
      router.push("/");
    }
  };

  const typeIcon: Record<string, string> = {
    paper: "📄",
    note: "📝",
    book: "📚",
    video: "🎥",
    webpage: "🌐"
  };

  return (
    <header id="header">
      <Link href="/" className="logo" onClick={() => handleNavClick("dashboard")}>
        <div className="logo-icon">🔬</div>
        <div>
          <div className="logo-text">Science Graph</div>
          <div className="logo-sub">Knowledge Explorer</div>
        </div>
      </Link>

      {/* Navigation Bar */}
      <nav className="header-nav">
        <button 
          className={`nav-btn ${pathname === "/" && activeView === "dashboard" ? "active" : ""}`}
          onClick={() => handleNavClick("dashboard")}
        >
          📊 Главная
        </button>
        <button 
          className={`nav-btn ${pathname === "/" && activeView === "graph" ? "active" : ""}`}
          onClick={() => handleNavClick("graph")}
        >
          🗺️ Граф
        </button>
        <button 
          className={`nav-btn ${pathname === "/" && activeView === "chat" ? "active" : ""}`}
          onClick={() => handleNavClick("chat")}
        >
          💬 Чат
        </button>
        <button 
          className={`nav-btn ${pathname === "/" && activeView === "notes" ? "active" : ""}`}
          onClick={() => handleNavClick("notes")}
        >
          📝 Заметки
        </button>
        <button 
          className={`nav-btn ${pathname === "/" && activeView === "chronology" ? "active" : ""}`}
          onClick={() => handleNavClick("chronology")}
        >
          ⏳ Временная шкала
        </button>
        <button 
          className={`nav-btn ${pathname === "/" && activeView === "upload" ? "active" : ""}`}
          onClick={() => handleNavClick("upload")}
        >
          📤 Загрузка
        </button>
        <button 
          className={`nav-btn ${pathname === "/library" ? "active" : ""}`}
          onClick={() => handleNavClick("library")}
        >
          📚 Библиотека
        </button>
      </nav>

      {/* Global Search wrapper with debounce dropdown */}
      <div className="search-wrap">
        <span className="search-icon">🔍</span>
        <input 
          type="text" 
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onFocus={() => setShowResults(searchResults.length > 0)}
          onBlur={() => setTimeout(() => setShowResults(false), 200)}
          placeholder="Поиск по статьям, книгам, заметкам…" 
          autoComplete="off" 
        />
        {showResults && (
          <div id="search-results" className="visible">
            {searchResults.map((item) => (
              <div 
                key={item.id}
                className="search-item" 
                onClick={() => selectSearchResult(item.id)}
                style={{ display: "flex", gap: "10px", alignItems: "center" }}
              >
                <span>{typeIcon[item.source_type] || "📄"}</span>
                <div>
                  <div className="search-item-title">{item.title}</div>
                  <div className="search-item-meta">{item.year || ""}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="header-actions">
        <div className="stat-badge">
          <span className="dot" style={{ background: "var(--col-paper)" }}></span>
          <span>{stats?.indexed_papers ?? "—"}</span> статей
        </div>
        <div className="stat-badge">
          <span className="dot" style={{ background: "var(--col-concept)" }}></span>
          <span>{stats?.concepts ?? "—"}</span> концептов
        </div>
        <button className="btn btn-ghost" onClick={refreshAll} title="Обновить граф">
          ↺ Обновить
        </button>
      </div>
    </header>
  );
}
