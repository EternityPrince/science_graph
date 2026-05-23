"use client";

import { useEffect, useRef } from "react";
import { useStore } from "@/lib/store";
import { GraphNode, GraphEdge } from "@/lib/types";

export default function GraphCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const networkRef = useRef<any>(null);
  const nodesDatasetRef = useRef<any>(null);
  const edgesDatasetRef = useRef<any>(null);
  const nodesViewRef = useRef<any>(null);
  const edgesViewRef = useRef<any>(null);

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
  } = useStore();

  // Helper check for node visibility
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
    return isVisible;
  };

  const isEdgeVisible = (e: GraphEdge) => {
    if (!nodesDatasetRef.current) return false;
    const fromNode = nodesDatasetRef.current.get(e.from);
    const toNode = nodesDatasetRef.current.get(e.to);
    return fromNode && isNodeVisible(fromNode) && toNode && isNodeVisible(toNode);
  };

  // Initialize Network
  useEffect(() => {
    if (!containerRef.current || !graphData || graphData.nodes.length === 0) return;

    let destroyed = false;

    const initNetwork = async () => {
      const { Network, DataSet, DataView } = await import("vis-network/standalone");
      if (destroyed) return;

      const processedNodes = graphData.nodes.map(n => ({
        ...n,
        opacity: 1.0,
        font: { color: '#c9cde0', size: 12, face: 'Inter' }
      }));

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

      nodesDatasetRef.current = new DataSet(processedNodes);
      edgesDatasetRef.current = new DataSet(processedEdges);

      nodesViewRef.current = new DataView(nodesDatasetRef.current, { filter: isNodeVisible });
      edgesViewRef.current = new DataView(edgesDatasetRef.current, { filter: isEdgeVisible });

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
        { nodes: nodesViewRef.current, edges: edgesViewRef.current },
        options
      );

      networkRef.current = network;

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
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [graphData]);

  // Handle selection updates (highlight selected neighbors, dim rest)
  useEffect(() => {
    const network = networkRef.current;
    const nodesDs = nodesDatasetRef.current;
    const edgesDs = edgesDatasetRef.current;

    if (!network || !nodesDs || !edgesDs) return;

    if (!selectedNodeId) {
      // Reset Highlight
      const nodes = nodesDs.get();
      const nodeUpdates = nodes.map((n: any) => ({
        id: n.id,
        opacity: 1.0,
        font: { color: "#c9cde0" },
      }));
      nodesDs.update(nodeUpdates);

      const edges = edgesDs.get();
      const edgeUpdates = edges.map((e: any) => ({
        id: e.id,
        color: e.originalColor,
        font: { color: "rgba(201, 205, 224, 0.6)" },
      }));
      edgesDs.update(edgeUpdates);

      network.selectNodes([]);
      return;
    }

    // Select target node
    network.selectNodes([selectedNodeId]);

    // Center camera on focused node
    network.focus(selectedNodeId, {
      scale: 1.2,
      animation: { duration: 500, easingFunction: "easeInOutQuad" },
    });

    const connectedNodes = new Set<string>(network.getConnectedNodes(selectedNodeId));
    connectedNodes.add(selectedNodeId);

    const connectedEdges = new Set<string>(network.getConnectedEdges(selectedNodeId));

    // Update nodes visibility
    const nodes = nodesDs.get();
    const nodeUpdates = nodes.map((n: any) => {
      const isConnected = connectedNodes.has(n.id);
      return {
        id: n.id,
        opacity: isConnected ? 1.0 : 0.15,
        font: {
          color: isConnected ? "#c9cde0" : "rgba(201, 205, 224, 0.15)",
        },
      };
    });
    nodesDs.update(nodeUpdates);

    // Update edges visibility
    const edges = edgesDs.get();
    const edgeUpdates = edges.map((e: any) => {
      const isConnected = connectedEdges.has(e.id);
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
    edgesDs.update(edgeUpdates);
  }, [selectedNodeId]);

  // Synchronize options change
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

  // Toggle edge labels
  useEffect(() => {
    if (!edgesDatasetRef.current) return;
    const edges = edgesDatasetRef.current.get();
    const updates = edges.map((e: any) => {
      const originalLabel = e.originalLabel !== undefined ? e.originalLabel : (e.label || "");
      return {
        id: e.id,
        label: edgeLabels ? originalLabel : "",
      };
    });
    edgesDatasetRef.current.update(updates);
  }, [edgeLabels]);

  // Re-filter elements on active filters change
  useEffect(() => {
    if (nodesViewRef.current && edgesViewRef.current) {
      nodesViewRef.current.refresh();
      edgesViewRef.current.refresh();
    }
  }, [filters, heatmapDate, fromDate, toDate]);

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
