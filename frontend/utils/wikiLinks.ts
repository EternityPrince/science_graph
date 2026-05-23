import { slugify } from "./slugify";
import { GraphNode } from "../lib/types";

export function resolveWikiLink(target: string, nodes: GraphNode[]): string {
  const normalizedTarget = target.trim();
  
  // 1. Try exact ID match
  const exactMatch = nodes.find(n => n.id === normalizedTarget);
  if (exactMatch) return exactMatch.id;

  // 2. Case-insensitive title match
  const titleMatch = nodes.find(n => {
    const title = n.full_title || n.label || "";
    return title.toLowerCase() === normalizedTarget.toLowerCase();
  });
  if (titleMatch) return titleMatch.id;

  // 3. Fallback to slugified concept ID
  return slugify(normalizedTarget);
}

export function parseWikiLinks(text: string, nodes: GraphNode[]): string {
  if (!text) return "";
  return text.replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (match, p1, p2) => {
    const target = p1.trim();
    const alias = p2 ? p2.trim() : target;
    const resolvedId = resolveWikiLink(target, nodes);
    return `<a href="#" class="wiki-link" data-node-id="${resolvedId}">${alias}</a>`;
  });
}
