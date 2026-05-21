import { fetchGraph } from './api.js';
import { toast } from './ui.js';
import { escapeHtml } from './utils.js';

let network = null;
let allNodes = null;
let allEdges = null;

export const activeFilters = new Set(['paper', 'note', 'book', 'author', 'concept', 'tag']);
export let activeHeatmapDate = null;

let onNodeClickCallback = null;
let viewSwitcherCallback = null;

/**
 * Register a callback for when a node is clicked.
 */
export function onNodeClick(cb) {
  onNodeClickCallback = cb;
}

/**
 * Register a callback to switch tab views.
 */
export function registerViewSwitcher(cb) {
  viewSwitcherCallback = cb;
}

/**
 * Getter for allNodes DataSet.
 */
export function getAllNodes() {
  return allNodes;
}

/**
 * Getter for network instance.
 */
export function getNetwork() {
  return network;
}

/**
 * Load and render the graph.
 */
export async function loadGraph() {
  const loadingEl = document.getElementById('graph-loading');
  const emptyEl = document.getElementById('graph-empty');

  if (loadingEl) loadingEl.classList.remove('hidden');
  if (emptyEl) emptyEl.classList.remove('visible');

  try {
    const data = await fetchGraph();

    if (!data.nodes || data.nodes.length === 0) {
      if (loadingEl) loadingEl.classList.add('hidden');
      if (emptyEl) emptyEl.classList.add('visible');
      return;
    }

    // Initialize vis.DataSets
    allNodes = new vis.DataSet(data.nodes);
    allEdges = new vis.DataSet(data.edges);

    const container = document.getElementById('mynetwork');
    network = new vis.Network(container, { nodes: allNodes, edges: allEdges }, {
      nodes: {
        font: { color: '#c9cde0', size: 12, face: 'Inter' },
        borderWidth: 1.5,
        borderWidthSelected: 3,
      },
      edges: {
        smooth: { type: 'continuous', roundness: 0.3 },
        selectionWidth: 2,
      },
      physics: {
        barnesHut: {
          gravitationalConstant: -9000,
          centralGravity: 0.25,
          springLength: 220,
          springConstant: 0.04,
          damping: 0.12,
        },
        stabilization: { iterations: 200 },
      },
      interaction: {
        hover: true,
        tooltipDelay: 150,
        hideEdgesOnDrag: true,
      },
    });

    network.on('stabilizationIterationsDone', () => {
      if (loadingEl) loadingEl.classList.add('hidden');
      network.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
    });

    network.on('click', async ({ nodes }) => {
      if (!nodes.length) return;
      if (onNodeClickCallback) {
        await onNodeClickCallback(nodes[0]);
      }
    });

    network.on('hoverNode', () => { container.style.cursor = 'pointer'; });
    network.on('blurNode',  () => { container.style.cursor = 'default'; });

    // Render depending views
    renderHeatmap(data.nodes);
    renderTimeline();

  } catch (e) {
    if (loadingEl) loadingEl.classList.add('hidden');
    toast('Ошибка загрузки графа: ' + e.message, 'err');
  }
}

/**
 * Filter application logic.
 */
export function applyFilters() {
  if (!allNodes) return;
  const all = allNodes.get();

  const fromDate = document.getElementById('filter-from-date')?.value || '';
  const toDate = document.getElementById('filter-to-date')?.value || '';

  const updates = all.map(n => {
    let isVisible = activeFilters.has(n.group);

    if (isVisible && n.created_at && ['paper', 'note', 'book'].includes(n.group)) {
      const docDate = n.created_at.substring(0, 10); // YYYY-MM-DD
      if (activeHeatmapDate) {
        if (docDate !== activeHeatmapDate) {
          isVisible = false;
        }
      } else {
        if (fromDate && docDate < fromDate) isVisible = false;
        if (toDate && docDate > toDate) isVisible = false;
      }
    }

    return {
      id: n.id,
      hidden: !isVisible,
    };
  });
  allNodes.update(updates);
}

/**
 * Focus camera on a node.
 */
export function focusNode(nodeId) {
  if (network) {
    network.selectNodes([nodeId]);
    network.focus(nodeId, { scale: 1.4, animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
  }
}

/**
 * Focus node and show details sidebar.
 */
export async function focusAndDetails(nodeId) {
  if (viewSwitcherCallback) {
    viewSwitcherCallback('graph');
  }
  focusNode(nodeId);
  if (onNodeClickCallback) {
    await onNodeClickCallback(nodeId);
  }
}

// Attach globally for dynamic HTML onclick hooks
window.focusAndDetails = focusAndDetails;

/**
 * Renders the calendar heatmap.
 */
function renderHeatmap(nodes) {
  const grid = document.getElementById('heatmap-grid');
  if (!grid) return;
  grid.innerHTML = '';

  const dateCounts = {};
  nodes.forEach(n => {
    if (n.created_at) {
      const dt = n.created_at.substring(0, 10);
      dateCounts[dt] = (dateCounts[dt] || 0) + 1;
    }
  });

  const today = new Date();
  const start = new Date(today);
  start.setDate(today.getDate() - 370); // Fill the 53 weeks x 7 days grid

  for (let i = 0; i < 371; i++) {
    const d = new Date(start);
    d.setDate(start.getDate() + i);
    const dateStr = d.toISOString().substring(0, 10);
    const count = dateCounts[dateStr] || 0;

    let level = 0;
    if (count === 1) level = 1;
    else if (count === 2) level = 2;
    else if (count === 3) level = 3;
    else if (count >= 4) level = 4;

    const cell = document.createElement('div');
    cell.className = `heatmap-cell level-${level}`;
    cell.dataset.date = dateStr;

    const options = { day: 'numeric', month: 'short', year: 'numeric' };
    const formattedDate = d.toLocaleDateString('ru-RU', options);
    cell.title = `${formattedDate}: ${count} док.`;

    cell.addEventListener('click', () => {
      toggleHeatmapDate(dateStr, cell);
    });

    grid.appendChild(cell);
  }
}

/**
 * Toggle active date on heatmap.
 */
function toggleHeatmapDate(dateStr, cell) {
  const cells = document.querySelectorAll('.heatmap-cell');
  const filterMsg = document.getElementById('filter-status-msg');

  if (activeHeatmapDate === dateStr) {
    activeHeatmapDate = null;
    cell.style.outline = 'none';
    if (filterMsg) filterMsg.style.display = 'none';
    toast('Фильтр по дню сброшен', 'info');
  } else {
    activeHeatmapDate = dateStr;
    cells.forEach(c => c.style.outline = 'none');
    cell.style.outline = '2px solid var(--accent)';
    if (filterMsg) {
      filterMsg.textContent = `Выбран день: ${dateStr}`;
      filterMsg.style.display = 'block';
    }
    toast(`Фильтр по дню: ${dateStr}`, 'ok');
  }
  applyFilters();
  renderTimeline();
}

/**
 * Reset date filters helper.
 */
export function resetDateFilters() {
  const fromEl = document.getElementById('filter-from-date');
  const toEl = document.getElementById('filter-to-date');
  if (fromEl) fromEl.value = '';
  if (toEl) toEl.value = '';

  activeHeatmapDate = null;
  const cells = document.querySelectorAll('.heatmap-cell');
  cells.forEach(c => c.style.outline = 'none');

  const filterMsg = document.getElementById('filter-status-msg');
  if (filterMsg) filterMsg.style.display = 'none';

  applyFilters();
  renderTimeline();
  toast('Фильтры дат сброшены', 'info');
}

/**
 * Apply date filters helper.
 */
export function applyDateFiltersBtn() {
  activeHeatmapDate = null;
  const cells = document.querySelectorAll('.heatmap-cell');
  cells.forEach(c => c.style.outline = 'none');

  const filterMsg = document.getElementById('filter-status-msg');
  if (filterMsg) filterMsg.style.display = 'none';

  applyFilters();
  renderTimeline();

  const fromDate = document.getElementById('filter-from-date')?.value || '';
  const toDate = document.getElementById('filter-to-date')?.value || '';
  toast(`Фильтр применен: c ${fromDate || '...'} по ${toDate || '...'}`, 'ok');
}

// Bind to window for HTML inline calls if necessary
window.resetDateFilters = resetDateFilters;
window.applyDateFiltersBtn = applyDateFiltersBtn;

/**
 * Render chronology list of publications.
 */
export function renderTimeline() {
  const list = document.getElementById('timeline-list');
  if (!list || !allNodes) return;

  const docs = allNodes.get({
    filter: function(item) {
      return ['paper', 'note', 'book'].includes(item.group);
    }
  });

  // Sort by created_at descending
  docs.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));

  const fromDate = document.getElementById('filter-from-date')?.value || '';
  const toDate = document.getElementById('filter-to-date')?.value || '';

  const filteredDocs = docs.filter(n => {
    if (!n.created_at) return false;
    const docDate = n.created_at.substring(0, 10);
    if (activeHeatmapDate) {
      return docDate === activeHeatmapDate;
    }
    if (fromDate && docDate < fromDate) return false;
    if (toDate && docDate > toDate) return false;
    return true;
  });

  if (filteredDocs.length === 0) {
    list.innerHTML = `<div style="font-size:12px;color:var(--text3);text-align:center;padding:10px;">В выбранном диапазоне нет документов</div>`;
    return;
  }

  const typeIcon = { paper: '📄', note: '📝', book: '📚' };

  list.innerHTML = filteredDocs.map(d => {
    const dateStr = d.created_at ? d.created_at.substring(0, 16).replace('T', ' ') : '—';
    const label = d.label || d.id;
    return `
      <div class="details-card" style="margin-bottom:8px;padding:12px;cursor:pointer;border-color:var(--border2);" onclick="focusAndDetails('${escapeHtml(d.id)}')">
        <div style="display:flex;gap:6px;align-items:flex-start;margin-bottom:4px;">
          <span>${typeIcon[d.group] || '📄'}</span>
          <span style="font-size:13px;font-weight:600;color:var(--text);line-height:1.4;">${escapeHtml(label)}</span>
        </div>
        <div style="font-size:11px;color:var(--text3);text-align:right;">
          <span>${dateStr}</span>
        </div>
      </div>
    `;
  }).join('');
}
