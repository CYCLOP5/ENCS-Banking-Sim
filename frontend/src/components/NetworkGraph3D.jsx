import { useRef, useCallback, useMemo, useEffect, useState } from "react";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";

const STATUS_COLORS = {
  Default: "#ff2a6d",
  Distressed: "#ffaa00",
  Safe: "#05d5fa",
  CCP: "#ffd700",
};

/**
 * 3D Force-Directed network graph for the banking topology.
 *
 * Props:
 *   - graphData: { nodes: [...], links: [...] }
 *   - statusMap: Record<nodeId, 'Default' | 'Distressed' | 'Safe'>
 *   - height / width
 *   - onNodeClick
 */
export default function NetworkGraph3D({
  graphData,
  statusMap = {},
  width,
  height,
  onNodeClick,
}) {
  const fgRef = useRef();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Enrich nodes with colour/size
  const enriched = useMemo(() => {
    if (!graphData) return { nodes: [], links: [] };
    const nodes = graphData.nodes.map((n) => ({
      ...n,
      color:
        n.region === "Global"
          ? STATUS_COLORS.CCP
          : STATUS_COLORS[statusMap[n.id]] || STATUS_COLORS.Safe,
      val: Math.max(Math.log10(n.total_assets || 1e9) - 8, 1),
    }));
    return { nodes, links: graphData.links };
  }, [graphData, statusMap]);

  // Custom node object — soft sphere
  const nodeThreeObject = useCallback((node) => {
    const radius = (node.val || 1) * 1.5;
    const geo = new THREE.SphereGeometry(radius, 16, 16);
    const mat = new THREE.MeshPhongMaterial({
      color: node.color,
      transparent: true,
      opacity: 0.85,
      emissive: node.color,
      emissiveIntensity: 0.35,
    });
    return new THREE.Mesh(geo, mat);
  }, []);

  // Link particles (glowing data flow)
  const linkDirectionalParticles = 2;
  const linkDirectionalParticleWidth = 1.2;
  const linkDirectionalParticleSpeed = 0.004;

  // Zoom to fit after data loads
  useEffect(() => {
    if (fgRef.current && mounted && enriched.nodes.length > 0) {
      const timer = setTimeout(() => {
        fgRef.current.zoomToFit(600, 80);
      }, 800);
      return () => clearTimeout(timer);
    }
  }, [mounted, enriched.nodes.length]);

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
        `<div style="background:rgba(13,13,20,0.9);padding:8px 12px;border-radius:8px;border:1px solid rgba(255,255,255,0.1);font-family:Inter,sans-serif;font-size:12px;">
          <strong>${n.name}</strong><br/>
          <span style="color:#888">Region: ${n.region}</span><br/>
          <span style="color:#888">Assets: $${(n.total_assets / 1e9).toFixed(1)}B</span>
        </div>`
      }
      linkColor={() => "rgba(50,224,196,0.12)"}
      linkWidth={0.3}
      linkOpacity={0.15}
      linkDirectionalParticles={linkDirectionalParticles}
      linkDirectionalParticleWidth={linkDirectionalParticleWidth}
      linkDirectionalParticleSpeed={linkDirectionalParticleSpeed}
      linkDirectionalParticleColor={() => "#32e0c4"}
      onNodeClick={onNodeClick}
      enableNodeDrag={false}
      warmupTicks={60}
      cooldownTime={3000}
      d3AlphaDecay={0.04}
      d3VelocityDecay={0.3}
    />
  );
}
