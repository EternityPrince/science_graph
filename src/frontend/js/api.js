const API_BASE = '';

/**
 * Fetch database stats.
 */
export async function fetchStats() {
  const r = await fetch(`${API_BASE}/api/stats`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

/**
 * Fetch configured AI models.
 */
export async function fetchModels() {
  const r = await fetch(`${API_BASE}/api/models`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

/**
 * Fetch entire graph structure (nodes + edges).
 */
export async function fetchGraph(showReferences = false) {
  const r = await fetch(`${API_BASE}/api/graph?show_references=${showReferences}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

const paperDetailsCache = new Map();

/**
 * Fetch a node's details by its ID.
 */
export async function fetchPaperDetails(paperId) {
  if (paperDetailsCache.has(paperId)) {
    return paperDetailsCache.get(paperId);
  }
  const r = await fetch(`${API_BASE}/api/paper/${encodeURIComponent(paperId)}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await r.json();
  paperDetailsCache.set(paperId, data);
  return data;
}

/**
 * Search papers by title.
 */
export async function searchPapers(q, signal) {
  const r = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(q)}`, { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

/**
 * Request opening a local file on the host.
 */
export async function openLocalFile(filePath) {
  const r = await fetch(`${API_BASE}/api/open-file`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_path: filePath })
  });
  if (!r.ok) {
    const d = await r.json();
    throw new Error(d.detail || 'Не удалось открыть файл');
  }
  return r.json();
}

/**
 * Fetch all note documents.
 */
export async function fetchNotes() {
  const r = await fetch(`${API_BASE}/api/notes`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

/**
 * Save a new note document.
 */
export async function saveNote(noteData) {
  const r = await fetch(`${API_BASE}/api/notes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(noteData)
  });
  if (!r.ok) {
    const res = await r.json();
    throw new Error(res.detail || 'Ошибка создания заметки');
  }
  return r.json();
}

/**
 * Upload and index a file.
 */
export async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const r = await fetch(`${API_BASE}/api/upload`, {
    method: 'POST',
    body: formData
  });
  if (!r.ok) {
    const d = await r.json();
    throw new Error(d.detail || r.statusText);
  }
  return r.json();
}

/**
 * Initiates the RAG stream query.
 */
export async function postQuery(question, limit = 5, cloud = false) {
  const r = await fetch(`${API_BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, limit, cloud })
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r;
}

/**
 * Ingests a URL (webpage or YouTube).
 */
export async function indexUrl(url) {
  const r = await fetch(`${API_BASE}/api/index-url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url })
  });
  if (!r.ok) {
    const d = await r.json();
    throw new Error(d.detail || 'Не удалось проиндексировать URL');
  }
  return r.json();
}
