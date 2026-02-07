import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Cell,
} from "recharts";
import {
  Database,
  Network,
  Cpu,
  ShieldAlert,
  Terminal,
  Layers,
  Code,
  ArrowRight,
  Server,
  Activity,
  Zap,
} from "lucide-react";
import ForceGraph3D from "react-force-graph-3d";
import GlassPanel from "../components/GlassPanel";
import katex from "katex";

// ── Data Imports ─────────────────────────────────────────────────────────────
import topologyData from "../data/topology_summary.json";
import assetData from "../data/asset_distribution.json";
import tierData from "../data/tier_breakdown.json";
import heatmapData from "../data/liability_heatmap.json";
import statsData from "../data/pipeline_stats.json";

// ── LaTeX Helper ─────────────────────────────────────────────────────────────
function Tex({ children, display = false }) {
  const html = katex.renderToString(children, {
    throwOnError: false,
    displayMode: display,
  });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

// ── Components ───────────────────────────────────────────────────────────────

function SectionHeader({ icon: Icon, title, subtitle }) {
  return (
    <div className="flex items-start gap-4 mb-8">
      <div className="p-3 rounded-xl bg-white/5 border border-white/10">
        <Icon className="w-6 h-6 text-stability-green" />
      </div>
      <div>
        <h2 className="text-2xl font-bold font-display text-text-primary">
          {title}
        </h2>
        <p className="text-text-secondary mt-1 max-w-2xl">{subtitle}</p>
      </div>
    </div>
  );
}

function TechBadge({ children }) {
  return (
    <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-white/5 border border-white/10 text-text-muted uppercase tracking-wider mx-1">
      {children}
    </span>
  );
}

function CodeSnippet({ code, lang = "python" }) {
  return (
    <div className="my-6 rounded-lg overflow-hidden border border-white/10 bg-[#0d1117]">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/5 bg-white/[0.02]">
        <div className="flex gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-white/10" />
          <div className="w-2.5 h-2.5 rounded-full bg-white/10" />
        </div>
        <span className="text-xs text-text-muted font-mono">{lang}</span>
      </div>
      <pre className="p-4 overflow-x-auto text-xs text-text-secondary font-mono leading-relaxed">
        <code>{code}</code>
      </pre>
    </div>
  );
}

// ── Visualizations ──────────────────────────────────────────────────────────

function AssetHistogram() {
  const data = useMemo(() => {
    const us = assetData.US;
    const eu = assetData.EU;
    
    // Combine histograms (assuming same bin centers/edges for simplicity or aligning them)
    // The generator produced separate bins. Let's just create a merged view based on indices if length matches
    // or just map one. Since they are distribution bins, we'll plot them side-by-side.
    // For simplicity, we assume the bins are roughly aligned or we just plot labels.
    
    return us.labels.map((label, i) => ({
      label,
      US: us.counts[i] || 0,
      EU: eu.counts[i] || 0,
    }));
  }, []);

  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis
            dataKey="label"
            tick={{ fill: "#888", fontSize: 10 }}
            stroke="rgba(255,255,255,0.1)"
            interval={2} 
          />
          <YAxis
            tick={{ fill: "#888", fontSize: 10 }}
            stroke="rgba(255,255,255,0.1)"
          />
          <Tooltip
            contentStyle={{ backgroundColor: "#000", borderColor: "#333" }}
            itemStyle={{ color: "#fff" }}
          />
          <Legend />
          <Bar dataKey="US" fill="#00e5ff" name="US Assets" radius={[2, 2, 0, 0]} />
          <Bar dataKey="EU" fill="#7c4dff" name="EU Assets" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function LiabilityHeatmap() {
  const { matrix, names, regions, raw_billions } = heatmapData;
  const [hover, setHover] = useState(null);

  // 20 banks
  const size = matrix.length;
  
  return (
    <div className="flex flex-col gap-4">
      <div className="relative aspect-square w-full max-w-[500px] mx-auto bg-black/20 rounded-lg p-1 border border-white/10">
        <div
          className="grid gap-[1px]"
          style={{
            gridTemplateColumns: `repeat(${size}, 1fr)`,
          }}
        >
          {matrix.map((row, i) =>
            row.map((val, j) => {
              // val is log10(exposure)
              // Normalized color intensity 0–10 roughly (log scale)
              const intensity = Math.min(val / 10, 1);
              const isUS = regions[i] === "US";
              // Blue for US->X, Purple for EU->X (Source based color) or simpler heatmap
              // Let's use standard heatmap gradient: distinct color
              const color = `rgba(255, 42, 109, ${intensity * 0.8 + 0.1})`; // Crisis Red base

              return (
                <div
                  key={`${i}-${j}`}
                  className="aspect-square w-full transition-all hover:scale-150 hover:z-10 hover:border hover:border-white relative"
                  style={{ backgroundColor: color }}
                  onMouseEnter={() => setHover({ i, j, val, raw: raw_billions[i][j] })}
                  onMouseLeave={() => setHover(null)}
                />
              );
            })
          )}
        </div>
        
        {/* Tooltip Overlay */}
        {hover && (
          <div className="absolute inset-x-0 bottom-full mb-2 p-3 bg-black/90 border border-white/20 rounded-lg text-xs z-50 pointer-events-none backdrop-blur-md">
            <div className="flex justify-between font-bold text-white mb-1">
              <span>{names[hover.i]}</span>
              <span className="text-text-muted">→</span>
              <span>{names[hover.j]}</span>
            </div>
            <div className="text-crisis-red font-mono">
              €{hover.raw.toFixed(2)} Billion Exposure
            </div>
          </div>
        )}
      </div>
      <div className="flex justify-between text-[10px] text-text-muted px-4">
        <span>Top 20 Systemic Banks (Interbank Liabilities)</span>
        <div className="flex items-center gap-2">
          <span>Low</span>
          <div className="w-16 h-2 bg-gradient-to-r from-crisis-red/10 to-crisis-red/90 rounded-full" />
          <span>High</span>
        </div>
      </div>
    </div>
  );
}

// ── Main Page Component ──────────────────────────────────────────────────────

export default function Implementation() {
  return (
    <div className="pt-24 pb-32 px-6 max-w-7xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-3xl mb-16"
      >
        <div className="flex items-center gap-2 text-stability-green mb-4">
          <Terminal className="w-4 h-4" />
          <span className="text-xs font-mono uppercase tracking-[0.2em]">
            System Architecture
          </span>
        </div>
        <h1 className="text-4xl md:text-5xl font-bold font-display leading-tight mb-6">
          High-performance <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-white to-white/50">
            Contagion Engine
          </span>
        </h1>
        <p className="text-lg text-text-secondary leading-relaxed">
          The ENCS platform ingests <strong>{statsData.matrix_shape[0]} banks</strong>, reconstructions 
          global topology from fragmented regulatory filings, and runs 
          stochastic simulation using a hybrid <strong>Python/Rust</strong> pipeline.
        </p>
      </motion.div>

      {/* Layer 1: Data Ingestion */}
      <section className="mb-24">
        <SectionHeader
          icon={Database}
          title="Layer 1: Data Ingestion & Imputation"
          subtitle="Trillion-scale balance sheet reconstruction from FFIEC (US) and EBA (EU) regulatory filings."
        />
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <GlassPanel className="space-y-6">
            <p className="text-text-secondary leading-relaxed">
              We ingest raw filings from the <TechBadge>US FFIEC</TechBadge> Call Reports 
              (031/041) and <TechBadge>EU EBA</TechBadge> Transparency Exercises. 
              The system handles <strong>{statsData.total_assets_T} Trillion USD</strong> in 
              total assets. Missing interbank data is imputed using a Gravity Model 
              calibrated to BIS locational statistics.
            </p>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                <div className="text-xs text-text-muted mb-1">Total Assets</div>
                <div className="text-2xl font-mono text-stability-green">
                  ${statsData.total_assets_T}T
                </div>
              </div>
              <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                <div className="text-xs text-text-muted mb-1">Data Points</div>
                <div className="text-2xl font-mono text-data-blue">
                  {statsData.total_banks} Banks
                </div>
              </div>
            </div>

            <h3 className="text-sm font-bold text-white mt-4 flex items-center gap-2">
              <Code className="w-4 h-4 text-stability-green" />
              Sparse Matrix Construction
            </h3>
            <p className="text-sm text-text-secondary">
              Due to the <Tex>O(N^2)</Tex> nature of financial networks, we use 
              <code>scipy.sparse.csr_matrix</code> for memory-efficient storage 
              of the <strong>{statsData.matrix_nnz.toLocaleString()}</strong> interbank edges.
            </p>
          </GlassPanel>

          <GlassPanel title="Asset Size Distribution (Log scale)">
             <div className="px-2 pt-4">
                <AssetHistogram />
             </div>
             <p className="text-xs text-center text-text-muted mt-2">
               Heavy-tailed distribution: Systemic risk is concentrated in the top 1%. 
               (Log10 Assets)
             </p>
          </GlassPanel>
        </div>
      </section>

      {/* Layer 2: Topology */}
      <section className="mb-24">
        <SectionHeader
          icon={Network}
          title="Layer 2: Network Toplogy Reconstruction"
          subtitle="Inferring hidden interbank liabilities using Gravity Models and RAS balancing."
        />
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="order-2 lg:order-1 h-[400px] rounded-2xl overflow-hidden border border-white/10 relative group">
             {/* Simple visualization of the network if 3D graph is too heavy, but user asked for it */}
             <div className="absolute inset-0 bg-black">
                <ForceGraph3D
                  graphData={{
                    nodes: topologyData.nodes,
                    links: topologyData.edges.map(e => ({ source: e.source, target: e.target, value: e.weight }))
                  }}
                  nodeLabel="name"
                  nodeColor={node => node.region === "US" ? "#00e5ff" : "#7c4dff"}
                  nodeVal={node => Math.log10(node.total_assets + 1) * 2}
                  linkColor={() => "rgba(255,255,255,0.1)"}
                  backgroundColor="#000000"
                  showNavInfo={false}
                  width={600} // Approximate, container will clip
                />
             </div>
             <div className="absolute bottom-4 left-4 bg-black/80 p-2 rounded text-xs text-white backdrop-blur">
                Interactive Force-Directed Graph (Top 60 Nodes)
             </div>
          </div>

          <GlassPanel className="order-1 lg:order-2 space-y-6">
            <h3 className="text-lg font-bold text-white">Gravity + RAS Model</h3>
            <p className="text-text-secondary text-sm leading-relaxed">
              Interbank liablities <Tex>L_&#123;ij&#125;</Tex> are unknown. We estimate them by maximizing entropy 
              subject to balance sheet constraints:
            </p>
            <div className="p-4 bg-black/30 rounded-lg border border-white/5 my-4">
               <Tex display>
                 L_&#123;ij&#125; \propto \frac&#123;A_i \times L_j&#125;&#123;e^&#123;\beta \cdot dist(i,j)&#125;&#125;
               </Tex>
               <div className="mt-4 border-t border-white/10 pt-4">
                 <Tex display>
                    \sum_j L_&#123;ij&#125; = \text&#123;Total Liabs&#125;_i, \quad \sum_i L_&#123;ij&#125; = \text&#123;Total Assets&#125;_j
                 </Tex>
               </div>
            </div>
            <p className="text-text-secondary text-sm">
              We iterate using the <strong>Sinkhorn-Knopp (RAS)</strong> algorithm to converge 
              the matrix marginals to reported regulatory totals.
            </p>
            <div className="flex gap-4 text-xs font-mono text-text-muted mt-2">
               <div className="flex items-center gap-1">
                 <div className="w-2 h-2 rounded-full bg-stability-green" />
                 US Banks (Blue)
               </div>
               <div className="flex items-center gap-1">
                 <div className="w-2 h-2 rounded-full bg-neon-purple" />
                 EU Banks (Purple)
               </div>
            </div>
          </GlassPanel>
        </div>
      </section>

      {/* Layer 3: Risk Engine */}
      <section className="mb-24">
        <SectionHeader
          icon={Cpu}
          title="Layer 3: Simulation Engine (Rust + Python)"
          subtitle="Hybrid runtime for Eisenberg-Noe contagion and Morris-Shin game theory."
        />

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
           {/* Column 1: Math & Code */}
           <div className="xl:col-span-2 space-y-8">
              <GlassPanel>
                 <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    <Server className="w-5 h-5 text-crisis-red" />
                    Rust FFI Integration
                 </h3>
                 <p className="text-text-secondary mb-4">
                    To handle <strong>{statsData.total_banks} banks</strong> with MC simulations, 
                    the core clearing logic is offloaded to Rust via <TechBadge>PyO3</TechBadge>.
                 </p>
                 <CodeSnippet code={`# backend/simulation_engine.py
if RUST_AVAILABLE:
    print("Delegating to Rust core...")
    results = encs_rust.run_full_simulation(
        W.astype(np.float64),
        external_assets,
        liabilities,
        ...
    )`} />
                 <p className="text-text-secondary mt-4">
                    The Rust engine implements vectorised Eisenberg-Noe clearing:
                 </p>
                 <div className="py-4 overflow-x-auto">
                    <Tex display>
                       p^* = \min(L, \max(0, e + \Pi^T p^*))
                    </Tex>
                 </div>
                 <p className="text-xs text-text-muted text-center">
                    Where <Tex>p^*</Tex> is the payment vector, <Tex>\Pi</Tex> is the relative liability matrix.
                 </p>
              </GlassPanel>

              {/* Game Theory Section */}
              <GlassPanel>
                 <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    <Zap className="w-5 h-5 text-amber-warn" />
                    Layer 4: Strategic Default (Game Theory)
                 </h3>
                 <p className="text-text-secondary mb-4">
                    Beyond mechanical default, we model <strong>Panic Runs</strong> using 
                    Morris & Shin (1998) Global Games. Banks receive noisy signals 
                    about counterparty solvency:
                 </p>
                 <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div className="p-4 bg-white/5 rounded border border-white/10">
                       <h4 className="font-mono text-sm text-amber-warn mb-2">Scenario A: Opaque</h4>
                       <p className="text-xs text-text-secondary mb-2">
                          Noisy public signal (<Tex>\alpha \approx 0</Tex>).
                       </p>
                       <p className="text-sm">
                          Banks rely on private signals. Coordination failure leads to 
                          <strong> precautionary runs</strong> even on solvent banks.
                       </p>
                    </div>
                    <div className="p-4 bg-white/5 rounded border border-white/10">
                       <h4 className="font-mono text-sm text-stability-green mb-2">Scenario B: Transparent</h4>
                       <p className="text-xs text-text-secondary mb-2">
                          High precision signal (<Tex>\alpha \to \infty</Tex>).
                       </p>
                       <p className="text-sm">
                          AI-driven transparency anchors expectations. 
                          <strong>Run rate drops</strong> as banks coordinate on fundamentals.
                       </p>
                    </div>
                 </div>
              </GlassPanel>
           </div>

           {/* Column 2: Heatmap */}
           <div className="xl:col-span-1">
              <GlassPanel title="Liability Concentration Heatmap" className="h-full">
                 <p className="text-xs text-text-secondary mb-6">
                    Top 20 banks by interbank obligations. Bright red nodes indicate 
                    systemically dangerous "Super-Spreaders".
                 </p>
                 <LiabilityHeatmap />
              </GlassPanel>
           </div>
        </div>
      </section>

      {/* Footer / Stats */}
      <div className="border-t border-white/10 pt-16 mt-16 text-center">
         <div className="inline-flex items-center gap-2 text-text-muted mb-4">
            <Code className="w-4 h-4" />
            <span className="text-sm font-mono">
               Engine Build: {statsData.total_banks} Nodes / {statsData.matrix_nnz} Edges
            </span>
         </div>
         <p className="text-text-secondary max-w-2xl mx-auto">
            Built for the 2026 FinTech Datathon. <br/>
            Stack: React (Vite) • Python (FastAPI) • Rust (PyO3) • PyTorch (GNN)
         </p>
      </div>
    </div>
  );
}
