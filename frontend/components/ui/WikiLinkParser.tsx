"use client";

import React, { useMemo } from "react";
import { useRouter } from "next/navigation";
import { useStore } from "@/lib/store";
import { parseWikiLinks } from "@/utils/wikiLinks";
import { Marked } from "marked";
import DOMPurify from "dompurify";

interface WikiLinkParserProps {
  text?: string;
  className?: string;
}

export default function WikiLinkParser({ text = "", className = "" }: WikiLinkParserProps) {
  const router = useRouter();
  const { graphData, setView, setSelectedNodeId } = useStore();
  const graphNodes = useMemo(() => graphData?.nodes || [], [graphData]);

  const renderedContent = useMemo(() => {
    if (!text) return "";
    try {
      const markedInstance = new Marked();
      // Render markdown to HTML string
      const rawHtml = markedInstance.parse(text) as string;
      
      // Sanitize the HTML string (running only on client-side to prevent SSR window reference error)
      let cleanHtml = rawHtml;
      if (typeof window !== "undefined") {
        cleanHtml = DOMPurify.sanitize(rawHtml);
      }
      
      // Parse [[wiki links]] within the HTML
      return parseWikiLinks(cleanHtml, graphNodes);
    } catch (err) {
      console.error("Failed to parse markdown/wiki links:", err);
      return text;
    }
  }, [text, graphNodes]);

  const handleContainerClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement;
    // Check if target is a wiki-link
    if (target.classList.contains("wiki-link")) {
      e.preventDefault();
      const nodeId = target.getAttribute("data-node-id");
      if (nodeId) {
        setSelectedNodeId(nodeId);
        setView("graph");
        router.push("/");
      }
    }
  };

  return (
    <div
      className={`wiki-parsed-content ${className}`}
      onClick={handleContainerClick}
      dangerouslySetInnerHTML={{ __html: renderedContent }}
    />
  );
}
