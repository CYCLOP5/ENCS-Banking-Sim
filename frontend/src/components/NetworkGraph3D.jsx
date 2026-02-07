import { useRef, useCallback, useMemo, useEffect, useState, useImperativeHandle, forwardRef } from "react";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";
import { forceCollide } from "d3-force-3d";
import SpriteText from "three-spritetext";

/* ── Colour palettes ─────────────────────────────────────────────── */
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

/* ── Layout tuning constants ─────────────────────────────────────── */
const NODE_HARD_CAP   = 350;
const CHARGE_STRENGTH = -300;   // repulsive force (default d3 is -30, far too weak)
const CHARGE_MAX_DIST = 800;    // beyond this distance charge has no effect
const COLLISION_PAD   = 4;      // extra spacing around each node's collision radius
const LINK_DIST_HIGH  = 250;    // link distance for high-value edges (was 120)
const LINK_DIST_MED   = 400;    // link distance for medium-value edges (was 200)
const LINK_DIST_LOW   = 600;    // link distance for low-value edges (was 350)
const CENTER_STRENGTH = 0.04;   // gentle pull toward origin

/**
 * 3D Force-Directed network graph for the banking topology.
 *
 * PERF STRATEGY:
 *   • `enriched` (useMemo) only recomputes when graphData/statusMap/maxNodes change —
 *     NOT on every contagion animation tick.
 *   • Contagion visuals are applied imperatively: we mutate each node's cached THREE.Mesh
 *     material + scale directly, and read contagion state from refs (not React state).
 *   • Link callbacks also read from refs, so their Function identity is stable across
 *     contagion ticks — ForceGraph3D won't re-process links on every step.
 */
const NetworkGraph3D = forwardRef(function NetworkGraph3D({
  graphData,
  statusMap = {},
  contagionSet: contagionSetProp,
  contagionActive: contagionActiveProp = false,
  contagionLinks: contagionLinksProp,
  width,
  height,
  onNodeClick,
  maxNodes = Infinity,
}, ref) {
  const fgRef = useRef();
  const [mounted, setMounted] = useState(false);

  /* ── Contagion state lives in refs to avoid triggering enriched/nodeThreeObject ── */
  const contagionActiveRef  = useRef(contagionActiveProp);
  const contagionSetRef     = useRef(contagionSetProp);
  const contagionLinksRef   = useRef(contagionLinksProp);

  // Keep refs in sync with props — and imperatively repaint nodes on change
  useEffect(() => {
    contagionActiveRef.current = contagionActiveProp;
    contagionSetRef.current    = contagionSetProp;
    contagionLinksRef.current  = contagionLinksProp;

    // Imperatively update every cached mesh to reflect new contagion state
    applyContagionVisuals();
  }, [contagionActiveProp, contagionSetProp, contagionLinksProp]);

  useEffect(() => { setMounted(true); }, []);

  /* ── Geometry cache — shared across all nodes ── */
  const geometries = useMemo(() => ({
    large:  new THREE.SphereGeometry(1, 12, 12),
    medium: new THREE.SphereGeometry(1, 8, 8),
    small:  new THREE.SphereGeometry(1, 6, 6),
  }), []);

  /* ── Node mesh cache: Map<nodeId, THREE.Mesh> — lets us mutate in-place ── */
  const meshCacheRef = useRef(new Map());

  /* ═══════════════════════════════════════════════════════════════════════
     enriched — ONLY recomputes when graphData/statusMap/maxNodes change.
     Contagion props are deliberately EXCLUDED from deps.
     ═══════════════════════════════════════════════════════════════════════ */
  const enriched = useMemo(() => {
    if (!graphData) return { nodes: [], links: [] };

    // Deep clone to prevent force-graph from mutating source data in-place
    let rawNodes = structuredClone(graphData.nodes);
    let rawLinks = structuredClone(graphData.links);

    const hasStatus = Object.keys(statusMap).length > 0;

    // After simulation: hard-cap to NODE_HARD_CAP, prioritizing Default > Distressed > Safe
    if (hasStatus) {
      const defaults = [], distressed = [], safe = [];
      for (const n of rawNodes) {
        const st = statusMap[n.id];
        if (st === "Default") defaults.push(n);
        else if (st === "Distressed") distressed.push(n);
        else safe.push(n);
      }
      const byAssets = (a, b) => (b.total_assets || 0) - (a.total_assets || 0);
      defaults.sort(byAssets); distressed.sort(byAssets); safe.sort(byAssets);

      const picked = [];
      picked.push(...defaults.slice(0, NODE_HARD_CAP));
      const r1 = NODE_HARD_CAP - picked.length;
      if (r1 > 0) picked.push(...distressed.slice(0, r1));
      const r2 = NODE_HARD_CAP - picked.length;
      if (r2 > 0) picked.push(...safe.slice(0, r2));

      const kept = new Set(picked.map((n) => n.id));
      rawNodes = picked;
      rawLinks = rawLinks.filter(
        (l) => kept.has(l.source?.id ?? l.source) && kept.has(l.target?.id ?? l.target)
      );
    }
    // Pre-simulation lite mode
    else if (maxNodes < rawNodes.length) {
      const sorted = [...rawNodes].sort((a, b) => (b.total_assets || 0) - (a.total_assets || 0));
      const kept = new Set(sorted.slice(0, maxNodes).map((n) => n.id));
      rawNodes = sorted.slice(0, maxNodes);
      rawLinks = rawLinks.filter(
        (l) => kept.has(l.source?.id ?? l.source) && kept.has(l.target?.id ?? l.target)
      );
    }

    // Assign base colour/size (non-contagion state)
    const nodes = rawNodes.map((n) => {
      const val = Math.max(Math.log10(n.total_assets || 1e9) - 8, 0.8);
      let color;
      if (n.region === "Global") color = STATUS_COLORS.CCP;
      else if (hasStatus) color = STATUS_COLORS[statusMap[n.id]] || STATUS_COLORS.Safe;
      else color = REGION_COLORS[n.region] || REGION_COLORS.US;

      return { ...n, color, val, _baseVal: val, _opacity: 0.9, _emissiveIntensity: 0.25 };
    });

    // Clear mesh cache — new enriched data means ForceGraph will re-create nodes
    meshCacheRef.current.clear();

    return { nodes, links: rawLinks };
  }, [graphData, statusMap, maxNodes]);

  /* ═══════════════════════════════════════════════════════════════════════
     Imperative contagion visual update — mutates cached meshes directly.
     Called from the useEffect that syncs contagion props → refs.
     Zero React re-renders. O(N) at worst, typically O(delta) for new reveals.
     ═══════════════════════════════════════════════════════════════════════ */
  const applyContagionVisuals = useCallback(() => {
    const cache    = meshCacheRef.current;
    const active   = contagionActiveRef.current;
    const revealed = contagionSetRef.current;

    if (cache.size === 0) return; // meshes not yet created

    for (const [nodeId, mesh] of cache) {
      const node = mesh.__nodeData;
      if (!node) continue;

      const baseVal = node._baseVal ?? node.val ?? 1;
      let color, opacity, emissive, scale;

      if (active && revealed) {
        if (revealed.has(nodeId)) {
          const st = statusMap[nodeId];
          color = STATUS_COLORS[st] || STATUS_COLORS.Safe;
          const isHit = st === "Default" || st === "Distressed";
          if (isHit) {
            scale   = baseVal * 1.8 * 1.2;
            opacity = 1.0;
            emissive = 0.7;
          } else {
            scale   = baseVal * 0.45 * 1.2;
            opacity = 0.2;
            emissive = 0.1;
          }
        } else {
          color    = "#3c3c50";
          scale    = baseVal * 0.25 * 1.2;
          opacity  = 0.06;
          emissive = 0;
        }
      } else {
        // Normal (non-contagion) appearance
        color    = node.color;
        scale    = baseVal * 1.2;
        opacity  = 0.9;
        emissive = 0.25;
      }

      // Mutate material in-place (no new allocation)
      const mat = mesh.material;
      mat.color.set(color);
      mat.opacity  = opacity;
      mat.emissive.set(color);
      mat.emissiveIntensity = emissive;
      mat.depthWrite = opacity > 0.3;
      mat.needsUpdate = true;

      mesh.scale.setScalar(scale);
      mesh.renderOrder = opacity > 0.3 ? 1 : 0;
    }
  }, [statusMap]);

  /* ── Expose camera helpers to parent via ref ── */
  useImperativeHandle(ref, () => ({
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
    zoomToFit(duration = 800) {
      fgRef.current?.zoomToFit(duration, 80);
    },
    stopPhysics() {
      const fg = fgRef.current;
      if (!fg) return;
      const charge = fg.d3Force('charge');
      if (charge) charge.strength(0);
      const link = fg.d3Force('link');
      if (link) link.strength(0);
      fg.d3Force('center', null);
      fg.d3Force('collide', null);
    },
    pauseRendering()  { fgRef.current?.pauseAnimation(); },
    resumeRendering() { fgRef.current?.resumeAnimation(); },
  }), [enriched.nodes]);

  /* ── nodeThreeObject — stable deps (only geometries). Caches mesh per node. ── */
  const nodeThreeObject = useCallback((node) => {
    const radius = (node.val || 1) * 1.2;
    const geo = radius > 2.5 ? geometries.large : radius > 1.5 ? geometries.medium : geometries.small;
    const mat = new THREE.MeshLambertMaterial({
      color: node.color,
      transparent: true,
      opacity: node._opacity ?? 0.9,
      emissive: node.color,
      emissiveIntensity: node._emissiveIntensity ?? 0.25,
      depthWrite: (node._opacity ?? 0.9) > 0.3,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.renderOrder = (node._opacity ?? 0.9) > 0.3 ? 1 : 0;
    mesh.scale.setScalar(radius);

    // Store ref for imperative contagion updates
    mesh.__nodeData = node;
    meshCacheRef.current.set(node.id, mesh);

    return mesh;
  }, [geometries]); // <-- NO contagionActive/contagionSet deps

  /* ═══════════════════════════════════════════════════════════════════════
     FORCE LAYOUT CONFIGURATION — runs once after mount and whenever
     enriched data changes. Fixes the "nodes too close" problem.
     ═══════════════════════════════════════════════════════════════════════ */
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg || !mounted || enriched.nodes.length === 0) return;

    // Charge: strong repulsion to spread nodes apart
    const charge = fg.d3Force('charge');
    if (charge) {
      charge.strength(CHARGE_STRENGTH);
      charge.distanceMax(CHARGE_MAX_DIST);
    }

    // Center: gentle pull to keep graph near origin
    const center = fg.d3Force('center');
    if (center) center.strength(CENTER_STRENGTH);

    // Collision: prevent node overlap
    fg.d3Force('collide', forceCollide((node) => {
      const r = (node.val || 1) * 1.5 + COLLISION_PAD;
      return r;
    }).iterations(2));

    // Reheat so new forces take effect
    fg.d3ReheatSimulation();

    // Zoom to fit after layout settles
    const timer = setTimeout(() => fg.zoomToFit(600, 80), 1200);
    return () => clearTimeout(timer);
  }, [mounted, enriched]);

  /* ── Particle filter (top 150 links by value) ── */
  const particleFilter = useMemo(() => {
    if (!enriched.links || enriched.links.length === 0) return new Set();
    const sorted = [...enriched.links].sort((a, b) => (b.value || 0) - (a.value || 0));
    return new Set(
      sorted.slice(0, Math.min(150, sorted.length))
        .map((l) => `${l.source?.id ?? l.source}-${l.target?.id ?? l.target}`)
    );
  }, [enriched.links]);

  /* ── Per-link random speed ── */
  const linkSpeedMap = useMemo(() => {
    const map = new Map();
    if (!enriched.links) return map;
    for (const l of enriched.links) {
      const key = `${l.source?.id ?? l.source}-${l.target?.id ?? l.target}`;
      map.set(key, 0.001 + Math.random() * 0.007);
    }
    return map;
  }, [enriched.links]);

  /* ═══════════════════════════════════════════════════════════════════════
     Link callbacks — read contagion state from REFS so the function
     identity stays stable across contagion ticks.
     ═══════════════════════════════════════════════════════════════════════ */
  const linkColor = useCallback((link) => {
    const src = link.source?.id ?? link.source;
    const tgt = link.target?.id ?? link.target;
    const key = `${src}-${tgt}`;
    const keyRev = `${tgt}-${src}`;
    const active = contagionActiveRef.current;
    const links  = contagionLinksRef.current;
    if (active && links) {
      if (links.has(key) || links.has(keyRev)) {
        const srcHit = statusMap[src] === "Default" || statusMap[src] === "Distressed";
        const tgtHit = statusMap[tgt] === "Default" || statusMap[tgt] === "Distressed";
        if (srcHit && tgtHit) return "rgba(255,42,109,0.7)";
        if (srcHit || tgtHit) return "rgba(255,140,0,0.5)";
        return "rgba(50,224,196,0.35)";
      }
      return "rgba(40,40,60,0.04)";
    }
    return particleFilter.has(key) ? "rgba(50,224,196,0.25)" : "rgba(50,224,196,0.08)";
  }, [statusMap, particleFilter]);

  const linkWidth = useCallback((link) => {
    const active = contagionActiveRef.current;
    const links  = contagionLinksRef.current;
    if (!active || !links) return 0.4;
    const src = link.source?.id ?? link.source;
    const tgt = link.target?.id ?? link.target;
    const key = `${src}-${tgt}`;
    const keyRev = `${tgt}-${src}`;
    if (links.has(key) || links.has(keyRev)) return 1.2;
    return 0.1;
  }, []);

  const linkParticles = useCallback((link) => {
    const src = link.source?.id ?? link.source;
    const tgt = link.target?.id ?? link.target;
    const key = `${src}-${tgt}`;
    const keyRev = `${tgt}-${src}`;
    const active = contagionActiveRef.current;
    const links  = contagionLinksRef.current;
    if (active && links) {
      if (links.has(key) || links.has(keyRev)) {
        const srcHit = statusMap[src] === "Default" || statusMap[src] === "Distressed";
        const tgtHit = statusMap[tgt] === "Default" || statusMap[tgt] === "Distressed";
        if (srcHit || tgtHit) return 5;
      }
      return 0;
    }
    return particleFilter.has(key) ? (1 + Math.floor(Math.random() * 2)) : 0;
  }, [statusMap, particleFilter]);

  const linkParticleWidth = useCallback((link) => {
    const active = contagionActiveRef.current;
    const links  = contagionLinksRef.current;
    if (!active || !links) return 0.8;
    const src = link.source?.id ?? link.source;
    const tgt = link.target?.id ?? link.target;
    const key = `${src}-${tgt}`;
    const keyRev = `${tgt}-${src}`;
    if (links.has(key) || links.has(keyRev)) return 3.0;
    return 0.8;
  }, []);

  const linkParticleSpeed = useCallback((link) => {
    if (contagionActiveRef.current) return 0.003;
    const key = `${link.source?.id ?? link.source}-${link.target?.id ?? link.target}`;
    return linkSpeedMap.get(key) || 0.003;
  }, [linkSpeedMap]);

  const linkParticleColor = useCallback((link) => {
    const active = contagionActiveRef.current;
    const links  = contagionLinksRef.current;
    if (!active || !links) return "#32e0c4";
    const src = link.source?.id ?? link.source;
    const tgt = link.target?.id ?? link.target;
    const key = `${src}-${tgt}`;
    const keyRev = `${tgt}-${src}`;
    if (links.has(key) || links.has(keyRev)) return "#ff2a6d";
    return "#32e0c4";
  }, []);

  const linkDistanceFn = useCallback((link) => {
    const val = link.value || 0;
    return val > 0.5 ? LINK_DIST_HIGH : val > 0.1 ? LINK_DIST_MED : LINK_DIST_LOW;
  }, []);

  /* ── Contagion label sprites (imperative, no React re-render) ── */
  const prevContagionRef = useRef(new Set());
  const labelTimers = useRef(new Map());
  const labelSprites = useRef(new Map());

  useEffect(() => {
    const active   = contagionActiveProp;
    const revealed = contagionSetProp;

    if (!active || !revealed || !fgRef.current) {
      // Cleanup labels
      labelSprites.current.forEach((sprite) => fgRef.current?.scene().remove(sprite));
      labelSprites.current.clear();
      labelTimers.current.forEach(clearTimeout);
      labelTimers.current.clear();
      prevContagionRef.current = new Set();
      return;
    }
    const prev = prevContagionRef.current;
    const newIds = [...revealed].filter((id) => !prev.has(id));
    prevContagionRef.current = new Set(revealed);

    for (const nid of newIds) {
      const st = statusMap[nid];
      if (st !== "Default" && st !== "Distressed") continue;
      const node = enriched.nodes.find((n) => n.id === nid);
      if (!node || node.x == null) continue;

      const label = new SpriteText(
        `${(node.name || "").slice(0, 22)}\n$${((node.total_assets || 0) / 1e9).toFixed(1)}B · ${st}`,
        3.5, st === "Default" ? "#ff2a6d" : "#ff8c00"
      );
      label.backgroundColor = "rgba(0,0,0,0.7)";
      label.borderRadius = 4;
      label.padding = [3, 5];
      label.position.set(node.x, (node.y || 0) + 8, node.z || 0);
      fgRef.current.scene().add(label);
      labelSprites.current.set(nid, label);

      const timer = setTimeout(() => {
        fgRef.current?.scene().remove(label);
        labelSprites.current.delete(nid);
        labelTimers.current.delete(nid);
      }, 2500);
      labelTimers.current.set(nid, timer);
    }
  }, [contagionSetProp, contagionActiveProp, statusMap, enriched.nodes]);

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
      linkColor={linkColor}
      linkWidth={linkWidth}
      linkOpacity={0.35}
      linkDirectionalParticles={linkParticles}
      linkDirectionalParticleWidth={linkParticleWidth}
      linkDirectionalParticleSpeed={linkParticleSpeed}
      linkDirectionalParticleColor={linkParticleColor}
      onNodeClick={onNodeClick}
      enableNodeDrag={false}
      warmupTicks={200}
      cooldownTime={10000}
      d3AlphaDecay={0.02}
      d3VelocityDecay={0.2}
      d3AlphaMin={0.001}
      dagMode={null}
      linkDistance={linkDistanceFn}
      nodeRelSize={1.5}
    />
  );
});

export default NetworkGraph3D;
