import { fetchNotes, saveNote } from './api.js';
import { toast } from './ui.js';
import { escapeHtml, escapeSingleQuotes, slugify } from './utils.js';
import { getAllNodes, loadGraph, focusAndDetails } from './graph.js';

let noteSavedCallbacks = [];

/**
 * Register callback to trigger when a note is successfully saved.
 */
export function onNoteSaved(cb) {
  noteSavedCallbacks.push(cb);
}

/**
 * Load and display the note list.
 */
export async function loadNotes() {
  const list = document.getElementById('notes-list');
  if (!list) return;

  try {
    const notes = await fetchNotes();
    renderNotesList(notes);
  } catch (e) {
    toast('Ошибка загрузки заметок: ' + e.message, 'err');
  }
}

/**
 * Render list of notes.
 */
function renderNotesList(notes) {
  const list = document.getElementById('notes-list');
  if (!list) return;

  if (!notes || notes.length === 0) {
    list.innerHTML = `<div style="font-size:12px;color:var(--text3);text-align:center;padding:10px;">Заметок пока нет</div>`;
    return;
  }

  list.innerHTML = notes.map(n => {
    const dateStr = n.created_at ? n.created_at.substring(0, 10) : '—';
    return `
      <div class="details-card" style="margin-bottom:8px;padding:12px;cursor:pointer;border-color:var(--border2);" onclick="focusAndDetails('${escapeHtml(n.id)}')">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
          <span style="font-size:13px;font-weight:600;color:var(--text);">${escapeHtml(n.title)}</span>
        </div>
        <div style="font-size:11px;color:var(--text3);display:flex;justify-content:space-between;">
          <span>${escapeHtml(n.authors.join(', ')) || 'Без автора'}</span>
          <span>${dateStr}</span>
        </div>
      </div>
    `;
  }).join('');
}

/**
 * Save note form submission handler.
 */
export async function saveNewNote() {
  const titleVal = document.getElementById('note-title')?.value.trim();
  const contentVal = document.getElementById('note-content')?.value.trim();
  const authorsRaw = document.getElementById('note-authors')?.value.trim();
  const tagsRaw = document.getElementById('note-tags')?.value.trim();

  if (!titleVal || !contentVal) {
    toast('Заполните название и содержание заметки', 'err');
    return;
  }

  const authors = authorsRaw ? authorsRaw.split(',').map(a => a.trim()) : [];
  const tags = tagsRaw ? tagsRaw.split(',').map(t => t.trim()) : [];

  const btn = document.querySelector('#note-form button[type="submit"]');
  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Сохранение...';
  }

  try {
    await saveNote({ title: titleVal, content: contentVal, authors, tags });
    toast('Заметка успешно создана', 'ok');

    const form = document.getElementById('note-form');
    if (form) form.reset();

    // Trigger registered callbacks to refresh stats/graph/notes
    for (const cb of noteSavedCallbacks) {
      try {
        await cb();
      } catch (err) {
        console.error("Callback execution error", err);
      }
    }
  } catch (e) {
    toast(`Ошибка: ${e.message}`, 'err');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = 'Сохранить заметку';
    }
  }
}

/**
 * Resolves a wiki-link target title or ID to a node ID in the graph.
 */
export function resolveWikiLink(target) {
  const allNodes = getAllNodes();
  if (!allNodes) return slugify(target);

  // 1. Try to find by exact ID match
  if (allNodes.get(target)) {
    return target;
  }

  // 2. Try to find by case-insensitive title or label match
  const matches = allNodes.get({
    filter: (item) => {
      const title = item.full_title || item.label || '';
      return title.toLowerCase() === target.toLowerCase();
    }
  });

  if (matches.length > 0) {
    return matches[0].id;
  }

  // 3. Fallback: slugified concept ID
  return slugify(target);
}

/**
 * Replaces [[WikiLink]] or [[WikiLink|Alias]] notation with focus HTML links.
 */
export function parseWikiLinks(text) {
  if (!text) return '';
  return text.replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (match, p1, p2) => {
    const target = p1.trim();
    const alias = p2 ? p2.trim() : target;
    const resolvedId = resolveWikiLink(target);
    return `<a href="#" onclick="focusAndDetails('${escapeSingleQuotes(resolvedId)}')">${escapeHtml(alias)}</a>`;
  });
}
