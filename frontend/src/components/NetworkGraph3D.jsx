import { useRef, useCallback, useMemo, useEffect, useState, useImperativeHandle, forwardRef } from "react";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";
import SpriteText from "three-spritetext";

const STATUS_COLORS = {
  Default: "#ff2a6d",
  Distressed: "#ff8c00",
  Safe: "#00e5ff",
  CCP: "#ffd700",
};

const REGION_COLORS = {
  US: "#00e5ff",
  EU: "#7c4dff",
  Global: "#ffd700",
};

/**
 * 3D Force-Directed network graph for the banking topology.
 */
const NODE_HARD_CAP = 350;

const NetworkGraph3D = forwardRef(function NetworkGraph3D({
  graphData,
  statusMap = {},
  contagionSet,
  contagionActive = false,
  contagionLinks,
  width,
  height,
  onNodeClick,
  maxNodes = Infinity,
}, ref) {
  const fgRef = useRef();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Geometry cache — shared across all nodes to save GPU memory
  const geometries = useMemo(() => ({
    large: new THREE.SphereGeometry(1, 12, 12),
    medium: new THREE.SphereGeometry(1, 8, 8),
    small: new THREE.SphereGeometry(1, 6, 6),
  }), []);

  // Enrich nodes with colour/size — and trim to maxNodes if needed
  const enriched = useMemo(() => {
    if (!graphData) return { nodes: [], links: [] };

    // Deep clone to prevent force-graph from mutating source data in-place
    let rawNodes = structuredClone(graphData.nodes);
    let rawLinks = structuredClone(graphData.links);

    const hasStatus = Object.keys(statusMap).length > 0;

    // After simulation: hard-cap to NODE_HARD_CAP, prioritizing Default > Distressed > Safe
    if (hasStatus) {
      const defaults = [];
      const distressed = [];
      const safe = [];
      for (const n of rawNodes) {
        const st = statusMap[n.id];
        if (st === "Default") defaults.push(n);
        else if (st === "Distressed") distressed.push(n);
        else safe.push(n);
      }
      // Sort each bucket by total_assets desc so we keep the biggest
      const byAssets = (a, b) => (b.total_assets || 0) - (a.total_assets || 0);
      defaults.sort(byAssets);
      distressed.sort(byAssets);
      safe.sort(byAssets);

      const cap = NODE_HARD_CAP;
      const picked = [];
      // Take all defaults first (up to cap)
      picked.push(...defaults.slice(0, cap));
      // Fill remaining with distressed
      const remaining1 = cap - picked.length;
      if (remaining1 > 0) picked.push(...distressed.slice(0, remaining1));
      // Fill remaining with safe (green)
      const remaining2 = cap - picked.length;
      if (remaining2 > 0) picked.push(...safe.slice(0, remaining2));

      const kept = new Set(picked.map((n) => n.id));
      rawNodes = picked;
      rawLinks = rawLinks.filter(
        (l) =>
          kept.has(l.source?.id ?? l.source) &&
          kept.has(l.target?.id ?? l.target)
      );
    }
    // Pre-simulation lite mode: keep only the top-N nodes by total_assets
    else if (maxNodes < rawNodes.length) {
      const sorted = [...rawNodes].sort(
        (a, b) => (b.total_assets || 0) - (a.total_assets || 0)
      );
      const kept = new Set(sorted.slice(0, maxNodes).map((n) => n.id));
      rawNodes = sorted.slice(0, maxNodes);
      rawLinks = rawLinks.filter(
        (l) =>
          kept.has(l.source?.id ?? l.source) &&
          kept.has(l.target?.id ?? l.target)
      );
    }

    const nodes = rawNodes.map((n) => {
      let val = Math.max(Math.log10(n.total_assets || 1e9) - 8, 0.8);
      let color;
      let _opacity = 0.9;
      let _emissiveIntensity = 0.25;

      // During contagion animation — only revealed nodes get their status color
      if (contagionActive && contagionSet) {
        if (contagionSet.has(n.id)) {
          const st = statusMap[n.id];
          color = STATUS_COLORS[st] || STATUS_COLORS.Safe;
          const isHit = st === "Default" || st === "Distressed";
          if (isHit) {
            // Affected nodes: large, bright, glowing
            val *= 1.8;
            _opacity = 1.0;
            _emissiveIntensity = 0.7;
          } else {
            // Revealed safe nodes: smaller & semi-transparent
            val *= 0.45;
            _opacity = 0.2;
            _emissiveIntensity = 0.1;
          }
        } else {
          // Not yet revealed — nearly invisible
          color = "rgba(60,60,80,0.6)";
          val *= 0.25;
          _opacity = 0.06;
          _emissiveIntensity = 0;
        }
      } else if (n.region === "Global") {
        color = STATUS_COLORS.CCP;
      } else if (hasStatus) {
        color = STATUS_COLORS[statusMap[n.id]] || STATUS_COLORS.Safe;
      } else {
        color = REGION_COLORS[n.region] || REGION_COLORS.US;
      }
      return { ...n, color, val, _opacity, _emissiveIntensity };
    });
    return { nodes, links: rawLinks };
  }, [graphData, statusMap, maxNodes, contagionActive, contagionSet]);

  // Expose camera helpers to parent via ref
  useImperativeHandle(ref, () => ({
    /** Smoothly move camera to look at a specific node id */
    focusNode(nodeId, distance = 120) {
      const fg = fgRef.current;
      if (!fg) return;
      const node = enriched.nodes.find((n) => n.id === nodeId);
      if (!node || node.x == null) return;
      fg.cameraPosition(
        { x: node.x, y: node.y, z: node.z + distance },
        { x: node.x, y: node.y, z: node.z },
        1200
      );
    },
    /** Zoom out to fit entire graph */
    zoomToFit(duration = 800) {
      fgRef.current?.zoomToFit(duration, 80);
    },
    /** Stop D3 physics so nodes freeze in place (prevents camera jitter) */
    stopPhysics() {
      const fg = fgRef.current;
      if (!fg) return;
      // Neuter all forces so the simulation has nothing to move
      const charge = fg.d3Force('charge');
      if (charge) charge.strength(0);
      const link = fg.d3Force('link');
      if (link) link.strength(0);
      fg.d3Force('center', null);
    },
    /** Pause the WebGL render loop (saves GPU when modal is open) */
    pauseRendering() {
      fgRef.current?.pauseAnimation();
    },
    /** Resume the WebGL render loop */
    resumeRendering() {
      fgRef.current?.resumeAnimation();
    },
  }), [enriched.nodes]);

  // Custom node object — shared geometry, instanced-friendly
  const nodeThreeObject = useCallback((node) => {
    const radius = (node.val || 1) * 1.2;
    const geo = radius > 2.5 ? geometries.large : radius > 1.5 ? geometries.medium : geometries.small;
    const mat = new THREE.MeshLambertMaterial({
      color: node.color,
      transparent: true,
      opacity: node._opacity ?? 0.9,
      emissive: node.color,
      emissiveIntensity: node._emissiveIntensity ?? 0.25,
      depthWrite: (node._opacity ?? 0.9) > 0.3, // transparent nodes don't block depth buffer
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.renderOrder = (node._opacity ?? 0.9) > 0.3 ? 1 : 0; // draw opaque first
    mesh.scale.setScalar(radius);
    return mesh;
  }, [geometries, contagionActive, contagionSet]);

  // Zoom to fit after data loads
  useEffect(() => {
    if (fgRef.current && mounted && enriched.nodes.length > 0) {
      const timer = setTimeout(() => {
        fgRef.current.zoomToFit(600, 80);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [mounted, enriched.nodes.length]);

  // Only show particles on top ~200 links (by value) to save GPU
  const particleFilter = useMemo(() => {
    if (!enriched.links || enriched.links.length === 0) return new Set();
    const sorted = [...enriched.links].sort((a, b) => (b.value || 0) - (a.value || 0));
    const topN = sorted.slice(0, Math.min(150, sorted.length));
    return new Set(topN.map((l) => `${l.source?.id ?? l.source}-${l.target?.id ?? l.target}`));
  }, [enriched.links]);

  // Per-link random speed so transactions fire from random places, not all in sync
  const linkSpeedMap = useMemo(() => {
    const map = new Map();
    if (!enriched.links) return map;
    for (const l of enriched.links) {
      const key = `${l.source?.id ?? l.source}-${l.target?.id ?? l.target}`;
      // Random speed between 0.001 and 0.008
      map.set(key, 0.001 + Math.random() * 0.007);
    }
    return map;
  }, [enriched.links]);

  // Track which nodes just appeared so we can show a temporary label
  const prevContagionRef = useRef(new Set());
  const labelTimers = useRef(new Map());    // nodeId -> timeout handle
  const labelSprites = useRef(new Map());   // nodeId -> SpriteText mesh

  // When contagionSet changes, detect newly revealed affected nodes and show label
  useEffect(() => {
    if (!contagionActive || !contagionSet || !fgRef.current) {
      // Cleanup all labels when contagion ends
      labelSprites.current.forEach((sprite) => {
        fgRef.current?.scene().remove(sprite);
      });
      labelSprites.current.clear();
      labelTimers.current.forEach(clearTimeout);
      labelTimers.current.clear();
      prevContagionRef.current = new Set();
      return;
    }
    const prev = prevContagionRef.current;
    const newIds = [...contagionSet].filter((id) => !prev.has(id));
    prevContagionRef.current = new Set(contagionSet);

    for (const nid of newIds) {
      const st = statusMap[nid];
      if (st !== "Default" && st !== "Distressed") continue;
      const node = enriched.nodes.find((n) => n.id === nid);
      if (!node || node.x == null) continue;

      const label = new SpriteText(
        `${(node.name || "").slice(0, 22)}\n$${((node.total_assets || 0) / 1e9).toFixed(1)}B · ${st}`,
        3.5,
        st === "Default" ? "#ff2a6d" : "#ff8c00"
      );
      label.backgroundColor = "rgba(0,0,0,0.7)";
      label.borderRadius = 4;
      label.padding = [3, 5];
      label.position.set(node.x, (node.y || 0) + 8, node.z || 0);
      fgRef.current.scene().add(label);
      labelSprites.current.set(nid, label);

      // Remove label after 2.5 seconds
      const timer = setTimeout(() => {
        fgRef.current?.scene().remove(label);
        labelSprites.current.delete(nid);
        labelTimers.current.delete(nid);
      }, 2500);
      labelTimers.current.set(nid, timer);
    }
  }, [contagionSet, contagionActive, statusMap, enriched.nodes]);

  if (!mounted) return null;

  return (
    <ForceGraph3D
      ref={fgRef}
      graphData={enriched}
      width={width}
      height={height}
      backgroundColor="rgba(0,0,0,0)"
      nodeThreeObject={nodeThreeObject}
      nodeLabel={(n) =>
        `<div style="background:rgba(10,10,18,0.92);padding:8px 14px;border-radius:10px;border:1px solid rgba(255,255,255,0.08);font-family:Inter,sans-serif;font-size:12px;line-height:1.5;backdrop-filter:blur(8px);">
          <strong style="color:#fff">${n.name}</strong><br/>
          <span style="color:${REGION_COLORS[n.region] || '#888'}">${n.region}</span> · 
          <span style="color:#999">$${(n.total_assets / 1e9).toFixed(1)}B</span>
        </div>`
      }
      linkColor={(link) => {
        const src = link.source?.id ?? link.source;
        const tgt = link.target?.id ?? link.target;
        const key = `${src}-${tgt}`;
        const keyRev = `${tgt}-${src}`;
        // During contagion, only show links that have been revealed
        if (contagionActive && contagionLinks) {
          if (contagionLinks.has(key) || contagionLinks.has(keyRev)) {
            const srcHit = statusMap[src] === "Default" || statusMap[src] === "Distressed";
            const tgtHit = statusMap[tgt] === "Default" || statusMap[tgt] === "Distressed";
            if (srcHit && tgtHit) return "rgba(255,42,109,0.7)";
            if (srcHit || tgtHit) return "rgba(255,140,0,0.5)";
            return "rgba(50,224,196,0.35)";
          }
          return "rgba(40,40,60,0.04)";
        }
        return particleFilter.has(key) ? "rgba(50,224,196,0.25)" : "rgba(50,224,196,0.08)";
      }}
      linkWidth={(link) => {
        if (!contagionActive || !contagionLinks) return 0.4;
        const src = link.source?.id ?? link.source;
        const tgt = link.target?.id ?? link.target;
        const key = `${src}-${tgt}`;
        const keyRev = `${tgt}-${src}`;
        if (contagionLinks.has(key) || contagionLinks.has(keyRev)) return 1.2;
        return 0.1;
      }}
      linkOpacity={0.35}
      linkDirectionalParticles={(link) => {
        const src = link.source?.id ?? link.source;
        const tgt = link.target?.id ?? link.target;
        const key = `${src}-${tgt}`;
        const keyRev = `${tgt}-${src}`;
        // During contagion, fire particles only along revealed links
        if (contagionActive && contagionLinks) {
          if (contagionLinks.has(key) || contagionLinks.has(keyRev)) {
            const srcHit = (statusMap[src] === "Default" || statusMap[src] === "Distressed");
            const tgtHit = (statusMap[tgt] === "Default" || statusMap[tgt] === "Distressed");
            if (srcHit || tgtHit) return 5;
          }
          return 0;
        }
        return particleFilter.has(key) ? (1 + Math.floor(Math.random() * 2)) : 0;
      }}
      linkDirectionalParticleWidth={(link) => {
        if (!contagionActive || !contagionLinks) return 0.8;
        const src = link.source?.id ?? link.source;
        const tgt = link.target?.id ?? link.target;
        const key = `${src}-${tgt}`;
        const keyRev = `${tgt}-${src}`;
        if (contagionLinks.has(key) || contagionLinks.has(keyRev)) return 3.0;
        return 0.8;
      }}
      linkDirectionalParticleSpeed={(link) => {
        if (contagionActive) return 0.003;
        const key = `${link.source?.id ?? link.source}-${link.target?.id ?? link.target}`;
        return linkSpeedMap.get(key) || 0.003;
      }}
      linkDirectionalParticleColor={(link) => {
        if (!contagionActive || !contagionLinks) return "#32e0c4";
        const src = link.source?.id ?? link.source;
        const tgt = link.target?.id ?? link.target;
        const key = `${src}-${tgt}`;
        const keyRev = `${tgt}-${src}`;
        if (contagionLinks.has(key) || contagionLinks.has(keyRev)) {
          return "#ff2a6d";
        }
        return "#32e0c4";
      }}
      onNodeClick={onNodeClick}
      enableNodeDrag={false}
      warmupTicks={100}
      cooldownTime={8000}
      d3AlphaDecay={0.02}
      d3VelocityDecay={0.3}
      d3AlphaMin={0.001}
      dagMode={null}
      linkDistance={(link) => {
        const val = link.value || 0;
        return val > 0.5 ? 120 : val > 0.1 ? 200 : 350;
      }}
      nodeRelSize={1.5}
    />
  );
});

export default NetworkGraph3D;
