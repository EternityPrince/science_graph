import { fetchStats, openLocalFile, uploadFile, fetchPaperDetails, fetchNotes, indexUrl } from './api.js';
import { toast, log } from './ui.js';
import { escapeHtml, escapeSingleQuotes } from './utils.js';
import { loadGraph, onNodeClick, registerViewSwitcher as registerGraphViewSwitcher, focusAndDetails, getNetwork, activeFilters, applyFilters, getAllNodes, highlightNeighbors, expandNodeReferences, clearExpandedReferences } from './graph.js';
import { loadNotes, saveNewNote, onNoteSaved, parseWikiLinks } from './notes.js';
import { sendMessage, registerViewSwitcher as registerChatViewSwitcher, askAbout } from './chat.js';

// Setup view switching
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    switchView(btn.dataset.view);
  });
});

export function switchView(viewName) {
  document.querySelectorAll('.nav-btn').forEach(b => {
    if (b.dataset.view === viewName) {
      b.classList.add('active');
    } else {
      b.classList.remove('active');
    }
  });
  document.querySelectorAll('.main-view').forEach(v => {
    if (v.id === `view-${viewName}`) {
      v.classList.add('active');
    } else {
      v.classList.remove('active');
    }
  });
  
  // Redraw network when switching to graph tab to avoid vis-network rendering artifacts
  const network = getNetwork();
  if (viewName === 'graph' && network) {
    network.redraw();
  }
}

// Stats Loader
export async function loadStats() {
  try {
    const d = await fetchStats();
    const papersEl = document.getElementById('stat-papers');
    const conceptsEl = document.getElementById('stat-concepts');
    if (papersEl) papersEl.textContent = d.papers ?? '—';
    if (conceptsEl) conceptsEl.textContent = d.concepts ?? '—';

    // Bento Stats
    const bentoPapersEl = document.getElementById('bento-stat-papers');
    const bentoConceptsEl = document.getElementById('bento-stat-concepts');
    const bentoNotesEl = document.getElementById('bento-stat-notes');
    const bentoStorageEl = document.getElementById('bento-stat-storage');

    if (bentoPapersEl) bentoPapersEl.textContent = d.papers ?? '—';
    if (bentoConceptsEl) bentoConceptsEl.textContent = d.concepts ?? '—';

    try {
      const notes = await fetchNotes();
      if (bentoNotesEl) bentoNotesEl.textContent = notes.length;
    } catch (e) {
      if (bentoNotesEl) bentoNotesEl.textContent = '—';
    }

    if (bentoStorageEl) {
      if (d.storage && d.storage.total_size !== undefined) {
        const sizeMb = d.storage.total_size / (1024 * 1024);
        bentoStorageEl.textContent = sizeMb < 0.1 ? '<0.1 MB' : `${sizeMb.toFixed(1)} MB`;
      } else {
        bentoStorageEl.textContent = '—';
      }
    }
  } catch (e) {
    toast('Ошибка загрузки статистики: ' + e.message, 'err');
  }
}

// ── Search Autocomplete ──────────────────────────────────────────────────────
const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
let searchTimer = null;
let searchAbortController = null;

if (searchInput && searchResults) {
  searchInput.addEventListener('input', () => {
    clearTimeout(searchTimer);
    if (searchAbortController) {
      searchAbortController.abort();
      searchAbortController = null;
    }
    const q = searchInput.value.trim();
    if (!q) {
      searchResults.classList.remove('visible');
      return;
    }
    searchTimer = setTimeout(() => doSearch(q), 350);
  });

  searchInput.addEventListener('blur', () => {
    setTimeout(() => searchResults.classList.remove('visible'), 200);
  });
}

// Expose functions needed by inline onclick handlers in dynamically rendered HTML
window.focusAndDetails = focusAndDetails;
window.askAbout = askAbout;

import { searchPapers } from './api.js';

async function doSearch(q) {
  if (searchAbortController) {
    searchAbortController.abort();
  }
  searchAbortController = new AbortController();

  try {
    const d = await searchPapers(q, searchAbortController.signal);
    renderSearchResults(d.results || []);
  } catch (e) {
    if (e.name === 'AbortError') {
      return;
    }
    toast('Ошибка поиска: ' + e.message, 'err');
  }
}

function renderSearchResults(results) {
  if (!searchResults) return;
  if (!results.length) {
    searchResults.classList.remove('visible');
    return;
  }
  const typeIcon = { paper: '📄', note: '📝', book: '📚' };
  searchResults.innerHTML = results.map(item => `
    <div class="search-item" data-id="${escapeHtml(item.id)}" onclick="selectSearchResult('${escapeHtml(item.id)}')">
      <span>${typeIcon[item.source_type] || '📄'}</span>
      <div>
        <div class="search-item-title">${escapeHtml(item.title)}</div>
        <div class="search-item-meta">${item.year || ''}</div>
      </div>
    </div>
  `).join('');
  searchResults.classList.add('visible');
}

async function selectSearchResult(nodeId) {
  if (searchResults) searchResults.classList.remove('visible');
  if (searchInput) searchInput.value = '';
  await focusAndDetails(nodeId);
}

window.selectSearchResult = selectSearchResult;

// ── File Upload Drag & Drop ──────────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

if (dropZone && fileInput) {
  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });
  
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  });
  
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleUpload(fileInput.files[0]);
  });
}

async function handleUpload(file) {
  const logBox = document.getElementById('upload-log');
  if (logBox) logBox.innerHTML = '';
  
  log(`📤 Загрузка: ${file.name} (${(file.size/1024).toFixed(1)} KB)`, 'info');

  try {
    log('⏳ Индексация…', 'info');
    const d = await uploadFile(file);

    log(`✅ Успешно проиндексировано: ID = ${d.id}`, 'ok');
    toast(`✅ ${file.name} добавлен в базу знаний`, 'ok');

    // Refresh components
    setTimeout(async () => {
      await loadGraph();
      await loadStats();
      await loadNotes();
      await updateDashboardLists();
    }, 800);

  } catch (e) {
    log(`❌ Ошибка: ${e.message}`, 'err');
    toast(`Ошибка загрузки: ${e.message}`, 'err');
  }

  if (fileInput) fileInput.value = '';
}

// ── URL Ingestion ─────────────────────────────────────────────────────────────
const urlIngestInput = document.getElementById('url-ingest-input');
const btnUrlIngest = document.getElementById('btn-url-ingest');
const urlIngestLog = document.getElementById('url-ingest-log');

async function handleUrlIngest() {
  if (!urlIngestInput) return;
  const url = urlIngestInput.value.trim();
  if (!url) return;

  if (urlIngestLog) {
    urlIngestLog.innerHTML = `<span class="log-info">⏳ Индексируем ссылку...</span>`;
  }
  toast(`Индексируем URL: ${url}`, 'info');

  try {
    const res = await indexUrl(url);

    if (urlIngestLog) {
      urlIngestLog.innerHTML = `<span class="log-ok">✅ Успешно проиндексировано: ${escapeHtml(res.title || url)}</span>`;
    }
    toast(`✅ Ссылка успешно добавлена в базу`, 'ok');

    urlIngestInput.value = '';

    setTimeout(async () => {
      await loadGraph();
      await loadStats();
      await loadNotes();
      await updateDashboardLists();
    }, 800);
  } catch (e) {
    if (urlIngestLog) {
      urlIngestLog.innerHTML = `<span class="log-err">❌ Ошибка: ${escapeHtml(e.message)}</span>`;
    }
    toast(`Ошибка индексирования URL: ${e.message}`, 'err');
  }
}

if (btnUrlIngest && urlIngestInput) {
  btnUrlIngest.addEventListener('click', handleUrlIngest);
  urlIngestInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleUrlIngest();
    }
  });
}

// ── Filter Chips ─────────────────────────────────────────────────────────────
document.querySelectorAll('.filter-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const group = chip.dataset.group;
    if (activeFilters.has(group)) {
      activeFilters.delete(group);
      chip.classList.add('inactive');
      chip.classList.remove('active');
    } else {
      activeFilters.add(group);
      chip.classList.remove('inactive');
      chip.classList.add('active');
    }
    applyFilters();
  });
});

// Refresh button
const btnRefresh = document.getElementById('btn-refresh');
if (btnRefresh) {
  btnRefresh.addEventListener('click', async () => {
    await Promise.all([loadGraph(), loadStats(), loadNotes()]);
    await updateDashboardLists();
  });
}

// Note form submit listener
const noteForm = document.getElementById('note-form');
if (noteForm) {
  noteForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    await saveNewNote();
  });
}

// Chat Send Button & Input resize
const chatSend = document.getElementById('chat-send');
const chatInput = document.getElementById('chat-input');

if (chatSend && chatInput) {
  chatSend.addEventListener('click', sendMessage);
  chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
  });
}

// ── Inter-Module Registrations ───────────────────────────────────────────────
onNodeClick(showNodeDetails);
registerGraphViewSwitcher(switchView);
registerChatViewSwitcher(switchView);
onNoteSaved(async () => {
  await Promise.all([loadNotes(), loadGraph(), loadStats()]);
  await updateDashboardLists();
});

// Expose seekVideo globally to allow inline onclick handlers in transcript buttons to seek the iframe player
window.seekVideo = function(videoId, seconds) {
  const iframe = document.getElementById(`yt-player-${videoId}`);
  if (iframe) {
    iframe.src = `https://www.youtube.com/embed/${videoId}?start=${seconds}&autoplay=1&enablejsapi=1&rel=0`;
  }
};

/**
 * Format timestamps in text like [MM:SS] or [HH:MM:SS] as clickable buttons that seek the video.
 */
function formatTextWithTimestamps(htmlStr, videoId) {
  if (!htmlStr) return '';
  const regex = /\[(?:(\d{1,2}):)?(\d{2}):(\d{2})\]/g;
  return htmlStr.replace(regex, (match, hh, mm, ss) => {
    const hours = hh ? parseInt(hh, 10) : 0;
    const minutes = parseInt(mm, 10);
    const seconds = parseInt(ss, 10);
    const totalSeconds = hours * 3600 + minutes * 60 + seconds;
    return `<button class="timestamp-btn" onclick="window.seekVideo('${escapeSingleQuotes(videoId)}', ${totalSeconds})">${match}</button>`;
  });
}

// ── Node Details panel polymorphic rendering ──────────────────────────────────
export async function showNodeDetails(nodeId) {
  const panel = document.getElementById('details-panel');
  if (!panel) return;
  panel.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;gap:12px;color:var(--text3)"><div class="spinner" style="width:24px;height:24px;border-width:2px"></div> Загрузка…</div>`;

  try {
    const d = await fetchPaperDetails(nodeId);
    
    // Dynamically expand citations and cited_by references on the graph
    if (d.type === 'paper') {
      expandNodeReferences(nodeId, d.citations, d.cited_by);
    } else {
      clearExpandedReferences();
    }
    
    // Highlight neighbors again to include the newly added reference nodes
    highlightNeighbors(nodeId);

    renderDetails(panel, d);
  } catch (e) {
    panel.innerHTML = `<div class="details-empty"><div class="icon">⚠️</div><p>${e.message}</p></div>`;
  }
}

function renderDetails(panel, d) {
  let html = '';

  // ── Render polymorphic Author node ──
  if (d.type === 'author') {
    html = `
      <div class="details-card">
        <h3>
          <span class="details-badge badge-author">Автор</span>
          ${escapeHtml(d.name)}
        </h3>
        <div class="details-row" style="margin-top: 10px;">
          <div class="details-field">
            <div class="details-label">Биография / Описание</div>
            <div class="details-value" style="font-size:13px; color:var(--text2); line-height:1.5;">${escapeHtml(d.description)}</div>
          </div>
        </div>
      </div>`;

    if (d.papers?.length) {
      const typeIcon = { paper: '📄', note: '📝', book: '📚' };
      html += `<div class="details-card">
        <h3>📚 Работы автора</h3>
        <div class="details-row" style="margin-top: 10px; gap: 10px;">
          ${d.papers.map(p => `
            <div class="details-value" style="font-size:13px; cursor:pointer; display:flex; gap:6px; align-items:center;" onclick="focusAndDetails('${escapeHtml(p.id)}')">
              <span>${typeIcon[p.source_type] || '📄'}</span>
              <span style="color: var(--accent);">${escapeHtml(p.title)}</span>
            </div>
          `).join('')}
        </div>
      </div>`;
    }
    panel.innerHTML = html;
    return;
  }

  // ── Render polymorphic Concept node ──
  if (d.type === 'concept') {
    html = `
      <div class="details-card">
        <h3>
          <span class="details-badge badge-concept">Концепт</span>
          ${escapeHtml(d.name)}
        </h3>
        <div class="details-row" style="margin-top: 10px;">
          <div class="details-field">
            <div class="details-label">Описание</div>
            <div class="details-value" style="font-size:13px; color:var(--text2); line-height:1.5;">${escapeHtml(d.description)}</div>
          </div>
        </div>
      </div>`;

    if (d.papers?.length) {
      const typeIcon = { paper: '📄', note: '📝', book: '📚' };
      html += `<div class="details-card">
        <h3>📚 Упоминания в работах</h3>
        <div class="details-row" style="margin-top: 10px; gap: 10px;">
          ${d.papers.map(p => `
            <div class="details-value" style="font-size:13px; cursor:pointer; display:flex; gap:6px; align-items:center;" onclick="focusAndDetails('${escapeHtml(p.id)}')">
              <span>${typeIcon[p.source_type] || '📄'}</span>
              <span style="color: var(--accent);">${escapeHtml(p.title)}</span>
            </div>
          `).join('')}
        </div>
      </div>`;
    }

    if (d.related?.length) {
      html += `<div class="details-card">
        <h3>🏷️ Связанные теги</h3>
        <div class="tag-list" style="margin-top: 10px;">
          ${d.related.map(t => `
            <span class="tag" style="border-color: var(--col-tag); color: var(--col-tag);" onclick="focusAndDetails('${escapeHtml(t.id)}')">${escapeHtml(t.name)}</span>
          `).join('')}
        </div>
      </div>`;
    }
    panel.innerHTML = html;
    return;
  }

  // ── Render polymorphic Tag node ──
  if (d.type === 'tag') {
    html = `
      <div class="details-card">
        <h3>
          <span class="details-badge badge-tag">Тег</span>
          ${escapeHtml(d.name)}
        </h3>
        <div class="details-row" style="margin-top: 10px;">
          <div class="details-field">
            <div class="details-label">Описание</div>
            <div class="details-value" style="font-size:13px; color:var(--text2); line-height:1.5;">${escapeHtml(d.description)}</div>
          </div>
        </div>
      </div>`;

    if (d.papers?.length) {
      const typeIcon = { paper: '📄', note: '📝', book: '📚' };
      html += `<div class="details-card">
        <h3>📚 Работы с этим тегом</h3>
        <div class="details-row" style="margin-top: 10px; gap: 10px;">
          ${d.papers.map(p => `
            <div class="details-value" style="font-size:13px; cursor:pointer; display:flex; gap:6px; align-items:center;" onclick="focusAndDetails('${escapeHtml(p.id)}')">
              <span>${typeIcon[p.source_type] || '📄'}</span>
              <span style="color: var(--accent);">${escapeHtml(p.title)}</span>
            </div>
          `).join('')}
        </div>
      </div>`;
    }

    if (d.related?.length) {
      html += `<div class="details-card">
        <h3>🧠 Связанные концепты</h3>
        <div class="tag-list" style="margin-top: 10px;">
          ${d.related.map(c => `
            <span class="tag" onclick="focusAndDetails('${escapeHtml(c.id)}')">${escapeHtml(c.name)}</span>
          `).join('')}
        </div>
      </div>`;
    }
    panel.innerHTML = html;
    return;
  }

  // ── Render Video node with Tabs ──
  if (d.source_type === 'video') {
    const props = d.properties || {};
    const uploader = props.uploader || (d.authors && d.authors[0]) || 'Неизвестный автор';
    const duration = props.duration ? `${Math.floor(props.duration / 60)} мин ${props.duration % 60} сек` : '';
    const publishDate = props.publish_date ? props.publish_date.substring(0, 10) : '';

    html = `
      <div class="details-card">
        <h3>
          <span class="details-badge badge-video">Видео</span>
          ${escapeHtml(d.title)}
        </h3>
        ${props.video_id ? `
          <div class="video-player-container">
            <iframe 
              id="yt-player-${props.video_id}"
              src="https://www.youtube.com/embed/${props.video_id}?enablejsapi=1&rel=0" 
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
              allowfullscreen>
            </iframe>
          </div>
        ` : ''}
        <div class="details-row" style="margin-top: 10px;">
          ${uploader ? `<div class="details-field"><div class="details-label">Автор</div><div class="details-value">${escapeHtml(uploader)}</div></div>` : ''}
          ${publishDate ? `<div class="details-field"><div class="details-label">Дата</div><div class="details-value">${publishDate}</div></div>` : ''}
          ${duration ? `<div class="details-field"><div class="details-label">Длительность</div><div class="details-value">${duration}</div></div>` : ''}
          ${props.url ? `<div class="details-field"><div class="details-label">Ссылка</div><div class="details-value"><a href="${escapeHtml(props.url)}" target="_blank">Открыть YouTube ↗</a></div></div>` : ''}
        </div>
      </div>`;

    if (props.video_overview || props.video_themes || props.video_outline || props.transcript) {
      const overviewRaw = props.video_overview ? `<p style="font-size: 13px; color: var(--text2); line-height: 1.6;">${escapeHtml(props.video_overview).replace(/\n/g, '<br>')}</p>` : '<p style="color:var(--text3)">Нет обзора</p>';
      const overviewHtml = props.video_id ? formatTextWithTimestamps(overviewRaw, props.video_id) : overviewRaw;
      
      let themesHtml = '';
      if (props.video_themes && Array.isArray(props.video_themes) && props.video_themes.length) {
        const rawThemes = props.video_themes.map(t => `<div class="video-theme-item">${parseWikiLinks(escapeHtml(t))}</div>`).join('');
        themesHtml = props.video_id ? formatTextWithTimestamps(rawThemes, props.video_id) : rawThemes;
      } else {
        themesHtml = '<div style="color:var(--text3); font-size:12px;">Нет тем</div>';
      }

      let outlineHtml = '';
      if (props.video_outline && Array.isArray(props.video_outline) && props.video_outline.length) {
        const rawOutline = props.video_outline.map(o => `<div class="video-outline-item">${parseWikiLinks(escapeHtml(o))}</div>`).join('');
        outlineHtml = props.video_id ? formatTextWithTimestamps(rawOutline, props.video_id) : rawOutline;
      } else {
        outlineHtml = '<div style="color:var(--text3); font-size:12px;">Нет конспекта</div>';
      }

      const transcriptText = props.transcript || 'Транскрипт отсутствует';
      const escapedTranscript = escapeHtml(transcriptText);
      const formattedTranscript = props.video_id ? formatTextWithTimestamps(escapedTranscript, props.video_id) : escapedTranscript;
      
      html += `
        <div class="details-card" style="padding:0; overflow:hidden; display:flex; flex-direction:column;">
          <div class="sidebar-tabs">
            <button class="sidebar-tab-btn active" data-tab="overview">Обзор</button>
            <button class="sidebar-tab-btn" data-tab="themes">Темы</button>
            <button class="sidebar-tab-btn" data-tab="outline">Конспект</button>
            <button class="sidebar-tab-btn" data-tab="transcript">Транскрипт</button>
          </div>
          
          <div class="sidebar-tab-content active" data-tab="overview" style="padding: 16px;">
            ${overviewHtml}
          </div>
          <div class="sidebar-tab-content" data-tab="themes" style="padding: 16px; gap: 8px; display: none;">
            ${themesHtml}
          </div>
          <div class="sidebar-tab-content" data-tab="outline" style="padding: 16px; gap: 8px; display: none;">
            ${outlineHtml}
          </div>
          <div class="sidebar-tab-content" data-tab="transcript" style="padding: 16px; display: none;">
            <div class="video-transcript-box">${formattedTranscript}</div>
          </div>
        </div>
      `;
    } else if (d.summary) {
      const parsedMarkdown = marked.parse(d.summary);
      const sanitizedMarkdown = DOMPurify.sanitize(parsedMarkdown);
      const processedSummary = parseWikiLinks(sanitizedMarkdown);
      html += `<div class="details-card">
        <h3>🎥 Обзор видео</h3>
        <div style="font-size: 13px; color: var(--text2); line-height: 1.6; margin-top: 10px; max-height: 250px; overflow-y: auto; background: var(--surface3); padding: 10px 12px; border-radius: var(--radius-sm);">
          ${processedSummary}
        </div>
      </div>`;
    }
    
    if (d.concepts && d.concepts.length) {
      html += `<div class="details-card">
        <h3>🧠 Концепты</h3>
        <div class="tag-list" style="margin-top: 10px;">${d.concepts.map(c =>
          `<span class="tag" onclick="focusAndDetails('${escapeHtml(c.id)}')">${escapeHtml(c.name)}</span>`
        ).join('')}</div>
      </div>`;
    }
  
    if (d.tags && d.tags.length) {
      html += `<div class="details-card">
        <h3>🏷️ Теги</h3>
        <div class="tag-list" style="margin-top: 10px;">${d.tags.map(t =>
          `<span class="tag" style="border-color: var(--col-tag); color: var(--col-tag);" onclick="focusAndDetails('${escapeHtml(t.id)}')">${escapeHtml(t.name)}</span>`
        ).join('')}</div>
      </div>`;
    }

    html += `<button class="btn btn-primary" style="width:100%;margin-top:8px" onclick="askAbout('${escapeSingleQuotes(d.title)}')">
      💬 Спросить об этой работе
    </button>`;

    panel.innerHTML = html;

    const tabButtons = panel.querySelectorAll('.sidebar-tab-btn');
    const tabContents = panel.querySelectorAll('.sidebar-tab-content');
    tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        tabButtons.forEach(b => b.classList.toggle('active', b === btn));
        tabContents.forEach(c => {
          if (c.dataset.tab === tabName) {
            c.classList.add('active');
            c.style.display = 'flex';
          } else {
            c.classList.remove('active');
            c.style.display = 'none';
          }
        });
      });
    });
    return;
  }

  // ── Render Paper (default document) node ──
  const typeLabel = { paper: 'Статья', note: 'Заметка', book: 'Книга', webpage: 'Веб-страница', video: 'Видео' }[d.source_type] || 'Документ';
  const badgeClass = { paper: 'badge-paper', note: 'badge-note', book: 'badge-book', webpage: 'badge-webpage', video: 'badge-video' }[d.source_type] || 'badge-paper';

  html = `
    <div class="details-card">
      <h3>
        <span class="details-badge ${badgeClass}">${typeLabel}</span>
        ${escapeHtml(d.title)}
      </h3>
      <div class="details-row" style="margin-top: 10px;">
        ${d.authors?.length ? `<div class="details-field"><div class="details-label">Авторы</div><div class="details-value">${escapeHtml(d.authors.join(', '))}</div></div>` : ''}
        ${d.year ? `<div class="details-field"><div class="details-label">Год</div><div class="details-value">${d.year}</div></div>` : ''}
        ${d.doi ? `<div class="details-field"><div class="details-label">DOI</div><div class="details-value"><a href="https://doi.org/${d.doi}" target="_blank">${escapeHtml(d.doi)}</a></div></div>` : ''}
        ${d.created_at ? `<div class="details-field"><div class="details-label">Добавлен</div><div class="details-value">${d.created_at.substring(0, 16).replace('T', ' ')}</div></div>` : ''}
      </div>
    </div>`;

  if (d.abstract) {
    const processedBody = parseWikiLinks(escapeHtml(d.abstract));
    html += `<div class="details-card">
      <h3>📄 Аннотация</h3>
      <div class="abstract-text" style="margin-top: 10px;">${processedBody}</div>
    </div>`;
  }

  if (d.summary) {
    const parsedMarkdown = marked.parse(d.summary);
    const sanitizedMarkdown = DOMPurify.sanitize(parsedMarkdown);
    const processedSummary = parseWikiLinks(sanitizedMarkdown);
    html += `<div class="details-card">
      <h3>💡 Краткое содержание (LLM Summary)</h3>
      <div style="font-size: 13px; color: var(--text2); line-height: 1.6; margin-top: 10px; max-height: 250px; overflow-y: auto; background: var(--surface3); padding: 10px 12px; border-radius: var(--radius-sm);">
        ${processedSummary}
      </div>
    </div>`;
  }

  if (d.concepts?.length) {
    html += `<div class="details-card">
      <h3>🧠 Концепты</h3>
      <div class="tag-list" style="margin-top: 10px;">${d.concepts.map(c =>
        `<span class="tag" onclick="focusAndDetails('${escapeHtml(c.id)}')">${escapeHtml(c.name)}</span>`
      ).join('')}</div>
    </div>`;
  }

  if (d.tags?.length) {
    html += `<div class="details-card">
      <h3>🏷️ Теги</h3>
      <div class="tag-list" style="margin-top: 10px;">${d.tags.map(t =>
        `<span class="tag" style="border-color: var(--col-tag); color: var(--col-tag);" onclick="focusAndDetails('${escapeHtml(t.id)}')">${escapeHtml(t.name)}</span>`
      ).join('')}</div>
    </div>`;
  }

  if (d.citations?.length) {
    html += `<div class="details-card">
      <h3>📎 Цитирует (${d.citations.length})</h3>
      <div class="details-row" style="margin-top: 10px; gap: 8px;">${d.citations.map(t =>
        `<div class="details-value" style="font-size:12px;color:var(--accent);cursor:pointer;" onclick="focusAndDetails('${escapeHtml(t.id)}')">• ${escapeHtml(t.title)}</div>`
      ).join('')}</div>
    </div>`;
  }

  if (d.cited_by?.length) {
    html += `<div class="details-card">
      <h3>📌 Цитируется в (${d.cited_by.length})</h3>
      <div class="details-row" style="margin-top: 10px; gap: 8px;">${d.cited_by.map(t =>
        `<div class="details-value" style="font-size:12px;color:var(--accent);cursor:pointer;" onclick="focusAndDetails('${escapeHtml(t.id)}')">• ${escapeHtml(t.title)}</div>`
      ).join('')}</div>
    </div>`;
  }

  if (d.file_path) {
    html += `<button class="btn btn-ghost" style="width:100%;margin-top:8px;display:flex;align-items:center;justify-content:center;gap:6px;" onclick="openLocalFile('${escapeSingleQuotes(d.file_path)}')">
      📂 Открыть локальный файл
    </button>`;
  }

  html += `<button class="btn btn-primary" style="width:100%;margin-top:8px" onclick="askAbout('${escapeSingleQuotes(d.title)}')">
    💬 Спросить об этой работе
  </button>`;

  panel.innerHTML = html;
}

window.openLocalFile = async (filePath) => {
  try {
    const d = await openLocalFile(filePath);
    toast(`Открыт файл: ${d.message || filePath}`, 'ok');
  } catch (e) {
    toast(`Ошибка открытия файла: ${e.message}`, 'err');
  }
};

// ── Bento Dashboard Populator ────────────────────────────────────────────────
export async function updateDashboardLists() {
  const recentDocsList = document.getElementById('bento-recent-docs-list');
  const recentNotesList = document.getElementById('bento-recent-notes-list');
  const conceptsList = document.getElementById('bento-concepts-list');

  const nodesDataSet = getAllNodes();
  if (!nodesDataSet) return;

  const all = nodesDataSet.get();

  // 1. Recent Publications (group === 'paper' or group === 'book' or group === 'video' or group === 'webpage')
  if (recentDocsList) {
    const docs = all.filter(n => ['paper', 'book', 'video', 'webpage'].includes(n.group));
    docs.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
    const topDocs = docs.slice(0, 5);

    if (topDocs.length === 0) {
      recentDocsList.innerHTML = `<div style="font-size:12px;color:var(--text3);padding:10px;text-align:center;">Нет публикаций</div>`;
    } else {
      const typeIcon = { paper: '📄', book: '📚', video: '🎥', webpage: '🌐' };
      recentDocsList.innerHTML = topDocs.map(d => {
        const title = d.full_title || d.label || d.id;
        const dateStr = d.created_at ? d.created_at.substring(0, 10) : '—';
        return `
          <div class="bento-list-item" onclick="focusAndDetails('${escapeHtml(d.id)}')">
            <div class="bento-list-item-icon">${typeIcon[d.group] || '📄'}</div>
            <div class="bento-list-item-content">
              <div class="bento-list-item-title">${escapeHtml(title)}</div>
              <div class="bento-list-item-meta">${dateStr}</div>
            </div>
          </div>
        `;
      }).join('');
    }
  }

  // 2. Recent Notes (group === 'note')
  if (recentNotesList) {
    const notes = all.filter(n => n.group === 'note');
    notes.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
    const topNotes = notes.slice(0, 5);

    if (topNotes.length === 0) {
      recentNotesList.innerHTML = `<div style="font-size:12px;color:var(--text3);padding:10px;text-align:center;">Нет заметок</div>`;
    } else {
      recentNotesList.innerHTML = topNotes.map(d => {
        const title = d.full_title || d.label || d.id;
        const dateStr = d.created_at ? d.created_at.substring(0, 10) : '—';
        return `
          <div class="bento-list-item" onclick="focusAndDetails('${escapeHtml(d.id)}')">
            <div class="bento-list-item-icon">📝</div>
            <div class="bento-list-item-content">
              <div class="bento-list-item-title">${escapeHtml(title)}</div>
              <div class="bento-list-item-meta">${dateStr}</div>
            </div>
          </div>
        `;
      }).join('');
    }
  }

  // 3. Popular Concepts/Tags (group === 'concept' or group === 'tag')
  if (conceptsList) {
    const concepts = all.filter(n => ['concept', 'tag'].includes(n.group));
    concepts.sort((a, b) => a.label.localeCompare(b.label));
    const topConcepts = concepts.slice(0, 15);

    if (topConcepts.length === 0) {
      conceptsList.innerHTML = `<div style="font-size:12px;color:var(--text3);padding:10px;">Нет концептов</div>`;
    } else {
      conceptsList.innerHTML = topConcepts.map(c => {
        const styleStr = c.group === 'tag' ? 'border-color: var(--col-tag); color: var(--col-tag);' : 'border-color: var(--col-concept); color: var(--col-concept);';
        return `
          <span class="tag" style="${styleStr} cursor:pointer; margin: 4px;" onclick="focusAndDetails('${escapeHtml(c.id)}')">${escapeHtml(c.label)}</span>
        `;
      }).join('');
    }
  }
}

// ── Quick Ask RAG Integration ───────────────────────────────────────────────
const quickAskInput = document.getElementById('quick-ask-input');
const btnQuickAsk = document.getElementById('btn-quick-ask');

if (quickAskInput && btnQuickAsk) {
  const triggerQuickAsk = () => {
    const q = quickAskInput.value.trim();
    if (!q) return;
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
      chatInput.value = q;
      chatInput.style.height = 'auto';
      chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    }
    quickAskInput.value = '';
    switchView('chat');
    sendMessage();
  };

  btnQuickAsk.addEventListener('click', triggerQuickAsk);
  quickAskInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      triggerQuickAsk();
    }
  });
}

// ── Application Bootstrapping ────────────────────────────────────────────────
(async () => {
  switchView('dashboard');
  await Promise.all([loadStats(), loadGraph(), loadNotes()]);
  await updateDashboardLists();
})();
