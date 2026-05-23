"use client";

import { useStore } from "@/lib/store";
import AuthorDetails from "./AuthorDetails";
import ConceptDetails from "./ConceptDetails";
import PaperDetails from "./PaperDetails";
import VideoDetails from "./VideoDetails";

export default function Sidebar() {
  const selectedNodeId = useStore((state) => state.selectedNodeId);
  const selectedNodeDetails = useStore((state) => state.selectedNodeDetails);

  if (!selectedNodeId) {
    return (
      <aside id="details-sidebar">
        <div className="sidebar-header">📋 Детали узла</div>
        <div id="details-panel">
          <div className="details-empty">
            <div className="icon">👆</div>
            <p>Кликните на узел графа, чтобы увидеть детали</p>
          </div>
        </div>
      </aside>
    );
  }

  return (
    <aside id="details-sidebar">
      <div className="sidebar-header">📋 Детали узла</div>
      <div id="details-panel">
        {!selectedNodeDetails ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", gap: "12px", color: "var(--text3)" }}>
            <div className="spinner" style={{ width: "24px", height: "24px", borderWidth: "2px" }}></div>
            Загрузка…
          </div>
        ) : (
          <div>
            {selectedNodeDetails.type === "author" && (
              <AuthorDetails details={selectedNodeDetails as any} />
            )}
            {(selectedNodeDetails.type === "concept" || selectedNodeDetails.type === "tag") && (
              <ConceptDetails details={selectedNodeDetails as any} />
            )}
            {selectedNodeDetails.type === "paper" && (
              selectedNodeDetails.source_type === "video" ? (
                <VideoDetails details={selectedNodeDetails as any} />
              ) : (
                <PaperDetails details={selectedNodeDetails as any} />
              )
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
