"use client";

import { useEffect, useRef, useState } from "react";
import { useStore } from "@/lib/store";
import { GraphNode, GraphEdge } from "@/lib/types";

export default function GraphCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const networkRef = useRef<any>(null);
  const nodesDatasetRef = useRef<any>(null);
  const edgesDatasetRef = useRef<any>(null);
  const [isInitialized, setIsInitialized] = useState(false);

  const {
    graphData,
    filters,
    heatmapDate,
    fromDate,
    toDate,
    spacing,
    gravity,
    edgeLength,
    physicsEnabled,
    physicsSolver,
    edgeLabels,
    selectedNodeId,
    setSelectedNodeId,
    setPhysicsEnabled,
    maxNodeDegree,
  } = useStore();

  // 1. Initialize Network with empty datasets
  useEffect(() => {
    if (!containerRef.current || !graphData || graphData.nodes.length === 0) return;

    let destroyed = false;

    const initNetwork = async () => {
      const { Network, DataSet } = await import("vis-network/standalone");
      if (destroyed) return;

      nodesDatasetRef.current = new DataSet([]);
      edgesDatasetRef.current = new DataSet([]);

      const options = {
        nodes: {
          borderWidth: 1.5,
          borderWidthSelected: 3,
        },
        edges: {
          smooth: { enabled: true, type: 'continuous', roundness: 0.3 },
          selectionWidth: 2,
        },
        physics: {
          enabled: physicsEnabled,
          solver: physicsSolver,
          barnesHut: {
            gravitationalConstant: spacing,
            centralGravity: gravity,
            springLength: edgeLength,
            springConstant: 0.04,
            damping: 0.12,
            avoidOverlap: 1.0,
          },
          forceAtlas2Based: {
            gravitationalConstant: -150,
            centralGravity: 0.01,
            springLength: edgeLength,
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
          hideEdgesOnZoom: true,
        },
      };

      const network = new Network(
        containerRef.current!,
        { nodes: nodesDatasetRef.current, edges: edgesDatasetRef.current },
        options
      );

      networkRef.current = network;
      setIsInitialized(true);

      // Stabilization hook
      network.on("stabilized", () => {
        console.log("Graph stabilized, disabling physics.");
        network.setOptions({ physics: { enabled: false } });
        setPhysicsEnabled(false);
      });

      // Selection hook
      network.on("click", ({ nodes }: { nodes: string[] }) => {
        if (!nodes.length) {
          setSelectedNodeId(null);
          return;
        }
        setSelectedNodeId(nodes[0]);
      });

      network.on("hoverNode", () => {
        if (containerRef.current) containerRef.current.style.cursor = "pointer";
      });
      network.on("blurNode", () => {
        if (containerRef.current) containerRef.current.style.cursor = "default";
      });
    };

    initNetwork();

    return () => {
      destroyed = true;
      setIsInitialized(false);
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [graphData]);

  // 2. Synchronize filters, date, maxNodeDegree, selection, edge label toggle, and edge length updates
  useEffect(() => {
    if (!isInitialized || !graphData || !nodesDatasetRef.current || !edgesDatasetRef.current || !networkRef.current) return;

    const network = networkRef.current;

    // Count degrees for nodes from edges
    const degreesCount: Record<string, number> = {};
    graphData.edges.forEach(e => {
      degreesCount[e.from] = (degreesCount[e.from] || 0) + 1;
      degreesCount[e.to] = (degreesCount[e.to] || 0) + 1;
    });

    const connectedToSelected = new Set<string>();
    if (selectedNodeId) {
      connectedToSelected.add(selectedNodeId);
      graphData.edges.forEach((e) => {
        if (e.from === selectedNodeId) {
          connectedToSelected.add(e.to);
        } else if (e.to === selectedNodeId) {
          connectedToSelected.add(e.from);
        }
      });
    }

    const isNodeVisible = (n: GraphNode) => {
      let isVisible = filters.has(n.group);

      if (isVisible && n.created_at && ['paper', 'note', 'book', 'video', 'webpage', 'reference'].includes(n.group)) {
        const docDate = n.created_at.substring(0, 10); // YYYY-MM-DD
        if (heatmapDate) {
          if (docDate !== heatmapDate) {
            isVisible = false;
          }
        } else {
          if (fromDate && docDate < fromDate) isVisible = false;
          if (toDate && docDate > toDate) isVisible = false;
        }
      }

      const degree = degreesCount[n.id] || 0;
      if (isVisible && maxNodeDegree > 0 && degree > maxNodeDegree) {
        if (selectedNodeId && connectedToSelected.has(n.id)) {
          // Keep it visible if it's connected to the selected node
        } else {
          isVisible = false;
        }
      }

      return isVisible;
    };

    const processedNodes = graphData.nodes.map(n => ({
      ...n,
      degree: degreesCount[n.id] || 0,
      opacity: 1.0,
      font: { color: '#c9cde0', size: 12, face: 'Inter' }
    }));

    const visibleNodes = processedNodes.filter(isNodeVisible);
    const visibleNodeIds = new Set(visibleNodes.map(n => n.id));

    const processedEdges = graphData.edges.map(e => {
      const originalLabel = e.label || "";
      const originalColor = e.color || { color: "rgba(255, 255, 255, 0.15)", highlight: "#6366f1" };
      
      let customLength = edgeLength;
      if (originalLabel === "MENTIONS_CONCEPT") {
        customLength = edgeLength * 1.4;
      } else if (originalLabel === "HAS_TAG") {
        customLength = edgeLength * 1.5;
      } else if (originalLabel === "AUTHORED") {
        customLength = edgeLength * 1.1;
      } else if (originalLabel === "CITES") {
        customLength = edgeLength * 1.6;
      }

      return {
        ...e,
        id: e.id || `${e.from}-${e.to}-${originalLabel}`,
        originalLabel,
        label: edgeLabels ? originalLabel : "",
        originalColor,
        length: customLength,
        font: { color: 'rgba(201, 205, 224, 0.6)', size: 8, align: 'top' }
      };
    });

    const visibleEdges = processedEdges.filter(e => visibleNodeIds.has(e.from) && visibleNodeIds.has(e.to));

    // Clear and batch load Vis.js DataSet
    nodesDatasetRef.current.clear();
    nodesDatasetRef.current.add(visibleNodes);

    edgesDatasetRef.current.clear();
    edgesDatasetRef.current.add(visibleEdges);

    // Apply selection highlighting / dimming
    if (!selectedNodeId) {
      network.selectNodes([]);
    } else {
      if (visibleNodeIds.has(selectedNodeId)) {
        network.selectNodes([selectedNodeId]);

        // Focus camera
        network.focus(selectedNodeId, {
          scale: 1.2,
          animation: { duration: 500, easingFunction: "easeInOutQuad" },
        });

        const connectedNodes = new Set<string>(network.getConnectedNodes(selectedNodeId));
        connectedNodes.add(selectedNodeId);

        const connectedEdges = new Set<string>(network.getConnectedEdges(selectedNodeId));

        // Update nodes opacity
        const nodesUpdates = visibleNodes.map((n) => {
          const isConnected = connectedNodes.has(n.id);
          return {
            id: n.id,
            opacity: isConnected ? 1.0 : 0.15,
            font: {
              color: isConnected ? "#c9cde0" : "rgba(201, 205, 224, 0.15)",
            },
          };
        });
        nodesDatasetRef.current.update(nodesUpdates);

        // Update edges opacity
        const edgesUpdates = visibleEdges.map((e) => {
          const isConnected = connectedEdges.has(e.id || "");
          return {
            id: e.id,
            color: isConnected
              ? e.originalColor
              : { color: "rgba(173, 181, 189, 0.12)", highlight: "rgba(116, 192, 252, 0.12)" },
            font: {
              color: isConnected ? "rgba(201, 205, 224, 0.6)" : "rgba(201, 205, 224, 0.1)",
            },
          };
        });
        edgesDatasetRef.current.update(edgesUpdates);
      }
    }
  }, [isInitialized, graphData, filters, heatmapDate, fromDate, toDate, maxNodeDegree, selectedNodeId, edgeLength, edgeLabels]);

  // 3. Synchronize physics options change
  useEffect(() => {
    if (!networkRef.current) return;
    networkRef.current.setOptions({
      physics: {
        enabled: physicsEnabled,
        solver: physicsSolver,
        barnesHut: {
          gravitationalConstant: spacing,
          centralGravity: gravity,
          springLength: edgeLength,
        },
        forceAtlas2Based: {
          springLength: edgeLength,
        },
      },
    });
  }, [physicsEnabled, physicsSolver, spacing, gravity, edgeLength]);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      {(!graphData || graphData.nodes.length === 0) && (
        <div id="graph-loading">
          <div className="spinner"></div>
          <div className="loading-text">Загрузка графа знаний…</div>
        </div>
      )}
      <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
    </div>
  );
}
