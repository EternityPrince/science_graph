import { fetchGraph } from './api.js';
import { toast } from './ui.js';
import { escapeHtml } from './utils.js';

let network = null;
let allNodes = null;
let allEdges = null;

export const activeFilters = new Set(['paper', 'note', 'book', 'video', 'webpage', 'author', 'concept', 'tag']);
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

    // Check edge labels toggle value
    const edgeLabelsToggle = document.getElementById('toggle-edge-labels');
    const showEdgeLabels = edgeLabelsToggle ? edgeLabelsToggle.checked : true;

    // Process nodes to ensure they have default opacity and font style
    const processedNodes = data.nodes.map(n => ({
      ...n,
      opacity: 1.0,
      font: { color: '#c9cde0', size: 12, face: 'Inter' }
    }));

    // Process edges to save original label and color
    const processedEdges = data.edges.map(e => {
      const originalLabel = e.label || "";
      const originalColor = e.color || { color: "#adb5bd", highlight: "#74c0fc" };
      return {
        ...e,
        originalLabel: originalLabel,
        label: showEdgeLabels ? originalLabel : "",
        originalColor: originalColor,
        font: { color: 'rgba(201, 205, 224, 0.6)', size: 8, align: 'top' }
      };
    });

    // Initialize vis.DataSets
    allNodes = new vis.DataSet(processedNodes);
    allEdges = new vis.DataSet(processedEdges);

    const container = document.getElementById('mynetwork');
    
    // Check physics configuration
    const solverSelect = document.getElementById('physics-solver-select');
    const initialSolver = solverSelect ? solverSelect.value : 'barnesHut';
    
    const physicsToggle = document.getElementById('toggle-physics');
    const physicsEnabled = physicsToggle ? !physicsToggle.checked : true;

    const spacingRange = document.getElementById('node-spacing-range');
    const gravityRange = document.getElementById('gravity-range');
    const edgeLengthRange = document.getElementById('edge-length-range');

    const currentSpacing = spacingRange ? parseInt(spacingRange.value) : -15000;
    const currentGravity = gravityRange ? parseFloat(gravityRange.value) : 0.04;
    const currentEdgeLength = edgeLengthRange ? parseInt(edgeLengthRange.value) : 260;

    network = new vis.Network(container, { nodes: allNodes, edges: allEdges }, {
      nodes: {
        borderWidth: 1.5,
        borderWidthSelected: 3,
      },
      edges: {
        smooth: { type: 'continuous', roundness: 0.3 },
        selectionWidth: 2,
      },
      physics: {
        enabled: physicsEnabled,
        solver: initialSolver,
        barnesHut: {
          gravitationalConstant: currentSpacing,
          centralGravity: currentGravity,
          springLength: currentEdgeLength,
          springConstant: 0.04,
          damping: 0.12,
          avoidOverlap: 1.0,
        },
        forceAtlas2Based: {
          gravitationalConstant: -150,
          centralGravity: 0.01,
          springLength: currentEdgeLength,
          springConstant: 0.08,
          damping: 0.4,
          avoidOverlap: 1.0,
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
      if (!nodes.length) {
        resetHighlighting();
        return;
      }
      const selectedNodeId = nodes[0];
      highlightNeighbors(selectedNodeId);
      if (onNodeClickCallback) {
        await onNodeClickCallback(selectedNodeId);
      }
    });

    network.on('hoverNode', () => { container.style.cursor = 'pointer'; });
    network.on('blurNode',  () => { container.style.cursor = 'default'; });

    // Render depending views
    renderHeatmap(data.nodes);
    renderTimeline();
    setupGraphControls();

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
  resetHighlighting();
  
  const all = allNodes.get();
  const fromDate = document.getElementById('filter-from-date')?.value || '';
  const toDate = document.getElementById('filter-to-date')?.value || '';

  const updates = all.map(n => {
    let isVisible = activeFilters.has(n.group);

    if (isVisible && n.created_at && ['paper', 'note', 'book', 'video', 'webpage'].includes(n.group)) {
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
  highlightNeighbors(nodeId);
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
  const dGrid = document.getElementById('dashboard-heatmap-grid');
  if (!grid && !dGrid) return;
  if (grid) grid.innerHTML = '';
  if (dGrid) dGrid.innerHTML = '';

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

    const options = { day: 'numeric', month: 'short', year: 'numeric' };
    const formattedDate = d.toLocaleDateString('ru-RU', options);
    const titleText = `${formattedDate}: ${count} док.`;

    if (grid) {
      const cell = document.createElement('div');
      cell.className = `heatmap-cell level-${level}`;
      cell.dataset.date = dateStr;
      cell.title = titleText;
      cell.addEventListener('click', () => {
        toggleHeatmapDate(dateStr, cell);
      });
      grid.appendChild(cell);
    }

    if (dGrid) {
      const cell = document.createElement('div');
      cell.className = `heatmap-cell level-${level}`;
      cell.dataset.date = dateStr;
      cell.title = titleText;
      cell.addEventListener('click', () => {
        toggleHeatmapDate(dateStr, cell);
      });
      dGrid.appendChild(cell);
    }
  }

  // Restore highlight on load if activeHeatmapDate is set
  if (activeHeatmapDate) {
    document.querySelectorAll(`.heatmap-cell[data-date="${activeHeatmapDate}"]`).forEach(c => {
      c.style.outline = '2px solid var(--accent)';
      c.style.outlineOffset = '-1px';
    });
  }
}

/**
 * Toggle active date on heatmap.
 */
function toggleHeatmapDate(dateStr, cell) {
  const filterMsg = document.getElementById('filter-status-msg');

  if (activeHeatmapDate === dateStr) {
    activeHeatmapDate = null;
    document.querySelectorAll(`.heatmap-cell[data-date="${dateStr}"]`).forEach(c => c.style.outline = 'none');
    if (filterMsg) filterMsg.style.display = 'none';
    toast('Фильтр по дню сброшен', 'info');
  } else {
    activeHeatmapDate = dateStr;
    document.querySelectorAll('.heatmap-cell').forEach(c => c.style.outline = 'none');
    document.querySelectorAll(`.heatmap-cell[data-date="${dateStr}"]`).forEach(c => {
      c.style.outline = '2px solid var(--accent)';
      c.style.outlineOffset = '-1px';
    });
    if (filterMsg) {
      filterMsg.textContent = `Выбран день: ${dateStr}`;
      filterMsg.style.display = 'block';
    }
    toast(`Фильтр по дню: ${dateStr}`, 'ok');
  }
  applyFilters();
  renderTimeline();

  if (viewSwitcherCallback) {
    viewSwitcherCallback('chronology');
  }
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
      return ['paper', 'note', 'book', 'video', 'webpage'].includes(item.group);
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

  const typeIcon = { paper: '📄', note: '📝', book: '📚', video: '🎥', webpage: '🌐' };

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

/**
 * Toggle edge labels visibility.
 */
function toggleEdgeLabels(show) {
  if (!allEdges) return;
  const edges = allEdges.get();
  const updates = edges.map(e => {
    const originalLabel = e.originalLabel !== undefined ? e.originalLabel : (e.label || "");
    return {
      id: e.id,
      label: show ? originalLabel : "",
      originalLabel: originalLabel
    };
  });
  allEdges.update(updates);
}

/**
 * Highlights a node's immediate neighbors and dims everything else.
 */
function highlightNeighbors(selectedNodeId) {
  if (!network || !allNodes || !allEdges) return;

  const connectedNodes = new Set(network.getConnectedNodes(selectedNodeId));
  connectedNodes.add(selectedNodeId);

  const connectedEdges = new Set(network.getConnectedEdges(selectedNodeId));

  // Dim nodes
  const nodes = allNodes.get();
  const nodeUpdates = nodes.map(n => {
    const isConnected = connectedNodes.has(n.id);
    return {
      id: n.id,
      opacity: isConnected ? 1.0 : 0.15,
      font: {
        color: isConnected ? '#c9cde0' : 'rgba(201, 205, 224, 0.15)'
      }
    };
  });
  allNodes.update(nodeUpdates);

  // Dim edges
  const edges = allEdges.get();
  const edgeUpdates = edges.map(e => {
    const isConnected = connectedEdges.has(e.id);
    return {
      id: e.id,
      color: isConnected ? e.originalColor : { color: 'rgba(173, 181, 189, 0.12)', highlight: 'rgba(116, 192, 252, 0.12)' },
      font: {
        color: isConnected ? 'rgba(201, 205, 224, 0.6)' : 'rgba(201, 205, 224, 0.1)'
      }
    };
  });
  allEdges.update(edgeUpdates);
}

/**
 * Resets all node and edge highlighting back to normal.
 */
function resetHighlighting() {
  if (!allNodes || !allEdges) return;

  const nodes = allNodes.get();
  const nodeUpdates = nodes.map(n => ({
    id: n.id,
    opacity: 1.0,
    font: {
      color: '#c9cde0'
    }
  }));
  allNodes.update(nodeUpdates);

  const edges = allEdges.get();
  const edgeUpdates = edges.map(e => ({
    id: e.id,
    color: e.originalColor,
    font: {
      color: 'rgba(201, 205, 224, 0.6)'
    }
  }));
  allEdges.update(edgeUpdates);
}

let controlsInitialized = false;

/**
 * Binds all sliders, selectors, reset actions, and local search inputs inside settings panel.
 */
function setupGraphControls() {
  if (controlsInitialized) return;
  controlsInitialized = true;

  const btnToggleSettings = document.getElementById('btn-toggle-settings');
  const btnCloseSettings = document.getElementById('btn-close-settings');
  const graphSettings = document.getElementById('graph-settings');

  if (btnToggleSettings && graphSettings) {
    btnToggleSettings.addEventListener('click', (e) => {
      e.stopPropagation();
      graphSettings.classList.toggle('visible');
    });
  }

  if (btnCloseSettings && graphSettings) {
    btnCloseSettings.addEventListener('click', (e) => {
      e.stopPropagation();
      graphSettings.classList.remove('visible');
    });
  }

  if (graphSettings) {
    graphSettings.addEventListener('click', (e) => {
      e.stopPropagation();
    });
  }

  document.addEventListener('click', () => {
    if (graphSettings && graphSettings.classList.contains('visible')) {
      graphSettings.classList.remove('visible');
    }
  });

  const solverSelect = document.getElementById('physics-solver-select');
  if (solverSelect) {
    solverSelect.addEventListener('change', () => {
      if (!network) return;
      network.setOptions({
        physics: { solver: solverSelect.value }
      });
    });
  }

  const spacingRange = document.getElementById('node-spacing-range');
  const spacingVal = document.getElementById('node-spacing-val');
  if (spacingRange && spacingVal) {
    spacingRange.addEventListener('input', () => {
      const val = parseInt(spacingRange.value);
      spacingVal.textContent = val;
      if (!network) return;
      network.setOptions({
        physics: {
          barnesHut: { gravitationalConstant: val }
        }
      });
    });
  }

  const gravityRange = document.getElementById('gravity-range');
  const gravityVal = document.getElementById('gravity-val');
  if (gravityRange && gravityVal) {
    gravityRange.addEventListener('input', () => {
      const val = parseFloat(gravityRange.value);
      gravityVal.textContent = val.toFixed(3);
      if (!network) return;
      network.setOptions({
        physics: {
          barnesHut: { centralGravity: val }
        }
      });
    });
  }

  const edgeRange = document.getElementById('edge-length-range');
  const edgeVal = document.getElementById('edge-length-val');
  if (edgeRange && edgeVal) {
    edgeRange.addEventListener('input', () => {
      const val = parseInt(edgeRange.value);
      edgeVal.textContent = val;
      if (!network) return;
      network.setOptions({
        physics: {
          barnesHut: { springLength: val },
          forceAtlas2Based: { springLength: val }
        }
      });
    });
  }

  const physicsToggle = document.getElementById('toggle-physics');
  if (physicsToggle) {
    physicsToggle.addEventListener('change', () => {
      if (!network) return;
      network.setOptions({
        physics: { enabled: !physicsToggle.checked }
      });
    });
  }

  const edgeLabelsToggle = document.getElementById('toggle-edge-labels');
  if (edgeLabelsToggle) {
    edgeLabelsToggle.addEventListener('change', () => {
      toggleEdgeLabels(edgeLabelsToggle.checked);
    });
  }

  const btnReset = document.getElementById('btn-reset-layout');
  if (btnReset) {
    btnReset.addEventListener('click', () => {
      if (solverSelect) solverSelect.value = 'barnesHut';
      if (spacingRange && spacingVal) {
        spacingRange.value = -15000;
        spacingVal.textContent = -15000;
      }
      if (gravityRange && gravityVal) {
        gravityRange.value = 0.04;
        gravityVal.textContent = 0.04;
      }
      if (edgeRange && edgeVal) {
        edgeRange.value = 260;
        edgeVal.textContent = 260;
      }
      if (physicsToggle) physicsToggle.checked = false;
      if (edgeLabelsToggle) edgeLabelsToggle.checked = true;

      if (!network) return;
      network.setOptions({
        physics: {
          enabled: true,
          solver: 'barnesHut',
          barnesHut: {
            gravitationalConstant: -15000,
            centralGravity: 0.04,
            springLength: 260,
          },
          forceAtlas2Based: {
            springLength: 260,
          }
        }
      });
      toggleEdgeLabels(true);
      resetHighlighting();
    });
  }

  const searchInput = document.getElementById('graph-search-input');
  const searchResults = document.getElementById('graph-search-results');

  if (searchInput && searchResults) {
    searchInput.addEventListener('input', () => {
      const q = searchInput.value.trim().toLowerCase();
      if (!q) {
        searchResults.classList.remove('visible');
        searchResults.innerHTML = '';
        return;
      }
      if (!allNodes) return;
      const matching = allNodes.get({
        filter: (item) => {
          if (item.hidden) return false;
          const label = (item.label || '').toLowerCase();
          const id = (item.id || '').toLowerCase();
          return label.includes(q) || id.includes(q);
        }
      });

      if (matching.length === 0) {
        searchResults.innerHTML = `<div style="padding: 8px 12px; font-size: 12px; color: var(--text3);">Ничего не найдено</div>`;
      } else {
        const typeIcon = { paper: '📄', note: '📝', book: '📚', video: '🎥', webpage: '🌐', author: '👤', concept: '🧠', tag: '🏷️' };
        searchResults.innerHTML = matching.slice(0, 10).map(item => `
          <div class="graph-search-result-item" data-id="${escapeHtml(item.id)}">
            <span>${typeIcon[item.group] || '📌'}</span>
            <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(item.label || item.id)}</span>
          </div>
        `).join('');

        searchResults.querySelectorAll('.graph-search-result-item').forEach(item => {
          item.addEventListener('click', async (e) => {
            e.stopPropagation();
            const nodeId = item.dataset.id;
            searchInput.value = '';
            searchResults.classList.remove('visible');
            searchResults.innerHTML = '';
            
            await focusAndDetails(nodeId);
          });
        });
      }
      searchResults.classList.add('visible');
    });

    document.addEventListener('click', (e) => {
      if (e.target !== searchInput && e.target !== searchResults) {
        searchResults.classList.remove('visible');
      }
    });

    searchInput.addEventListener('click', (e) => e.stopPropagation());
    searchResults.addEventListener('click', (e) => e.stopPropagation());
  }
}

