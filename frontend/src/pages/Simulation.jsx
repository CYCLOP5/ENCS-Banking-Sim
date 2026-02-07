import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Play,
  Settings,
  BarChart3,
  CloudLightning,
  Gamepad2,
  Loader2,
  AlertTriangle,
  TrendingDown,
  DollarSign,
  Users,
  Zap,
  ChevronRight,
} from "lucide-react";
import GlassPanel from "../components/GlassPanel";
import MetricCard from "../components/MetricCard";
import NetworkGraph3D from "../components/NetworkGraph3D";
import { fetchTopology, runSimulation, runClimate, runGame } from "../services/api";
import { cn, formatUSD } from "../lib/utils";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";

/* ── Tab Button ────────────────────────────────────────────────── */
function TabBtn({ active, onClick, icon: Icon, label }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-all",
        active
          ? "bg-white/[0.07] text-white border border-border-bright"
          : "text-text-muted hover:text-text-secondary hover:bg-surface"
      )}
    >
      <Icon className="h-3.5 w-3.5" />
      {label}
    </button>
  );
}

/* ── Slider Input ──────────────────────────────────────────────── */
function Slider({ label, value, onChange, min, max, step = 0.01, suffix = "" }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-text-muted uppercase tracking-wider font-[family-name:var(--font-mono)]">
          {label}
        </span>
        <span className="text-xs text-white font-[family-name:var(--font-mono)] font-bold">
          {typeof value === "number" ? value.toFixed(step < 0.1 ? 2 : 0) : value}
          {suffix}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer
          bg-gradient-to-r from-stability-green/30 to-crisis-red/30
          [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4
          [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full
          [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:shadow-md
          [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-stability-green"
      />
    </div>
  );
}

/* ── Toggle Switch ─────────────────────────────────────────────── */
function Toggle({ label, checked, onChange, description }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <div>
        <span className="text-[11px] text-text-muted uppercase tracking-wider font-[family-name:var(--font-mono)]">
          {label}
        </span>
        {description && (
          <p className="text-[10px] text-text-muted/60 mt-0.5">{description}</p>
        )}
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={cn(
          "relative flex h-5 w-9 shrink-0 rounded-full transition-colors",
          checked ? "bg-stability-green" : "bg-white/10"
        )}
      >
        <span
          className={cn(
            "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform",
            checked ? "translate-x-4" : "translate-x-0.5"
          )}
        />
      </button>
    </div>
  );
}

/* ── Default Ticker ────────────────────────────────────────────── */
function DefaultTicker({ defaults = [] }) {
  if (defaults.length === 0) return null;
  const doubled = [...defaults, ...defaults];
  return (
    <div className="overflow-hidden whitespace-nowrap mask-gradient">
      <div className="animate-ticker inline-flex gap-6">
        {doubled.map((name, i) => (
          <span
            key={i}
            className="inline-flex items-center gap-1.5 text-xs font-[family-name:var(--font-mono)]"
          >
            <span className="h-1.5 w-1.5 rounded-full bg-crisis-red animate-pulse" />
            <span className="text-crisis-red">{name}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   SIMULATION DASHBOARD
   ═══════════════════════════════════════════════════════════════════ */

export default function Simulation() {
  // ── State ──
  const [tab, setTab] = useState("mechanical");
  const [topology, setTopology] = useState(null);
  const [loading, setLoading] = useState(true);
  const [simulating, setSimulating] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const containerRef = useRef(null);
  const [dims, setDims] = useState({ w: 800, h: 600 });

  // ── Controls ──
  const [severity, setSeverity] = useState(1.0);
  const [nSteps, setNSteps] = useState(10);
  const [panicRate, setPanicRate] = useState(0.1);
  const [fireSaleAlpha, setFireSaleAlpha] = useState(0.005);
  const [useCcp, setUseCcp] = useState(false);
  const [clearingRate, setClearingRate] = useState(0.5);
  // Climate
  const [carbonTax, setCarbonTax] = useState(0.5);
  const [greenSubsidy, setGreenSubsidy] = useState(0.1);
  // Game
  const [gameTransparent, setGameTransparent] = useState(false);
  const [gameSolvency, setGameSolvency] = useState(0.2);

  // ── Load topology ──
  useEffect(() => {
    fetchTopology()
      .then(setTopology)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // ── Resize observer ──
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDims({ w: width, h: height });
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // ── Build status map from results ──
  const statusMap = {};
  if (results?.status) {
    results.status.forEach((s, i) => {
      statusMap[i] = s;
    });
  }

  // ── Execute simulation ──
  const execute = useCallback(async () => {
    setSimulating(true);
    setError(null);
    try {
      let res;
      if (tab === "climate") {
        res = await runClimate({
          carbonTax,
          greenSubsidy,
          severity,
          nSteps,
        });
      } else if (tab === "strategic") {
        res = await runGame({
          trueSolvency: gameSolvency,
          nBanks: 20,
          nSteps: 5,
        });
      } else {
        res = await runSimulation({
          severity,
          nSteps,
          panicRate,
          fireSaleAlpha,
          useCcp,
          clearingRate,
        });
      }
      setResults(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setSimulating(false);
    }
  }, [
    tab, severity, nSteps, panicRate, fireSaleAlpha, useCcp, clearingRate,
    carbonTax, greenSubsidy, gameSolvency,
  ]);

  // ── Timeline chart data ──
  const timelineData =
    results?.equity_loss_timeline?.map((val, i) => ({
      step: i,
      loss: val / 1e9,
      defaults: results.defaults_timeline?.[i] ?? 0,
      price: results.price_timeline?.[i] ?? 1,
    })) ?? [];

  // ── Defaulted bank names ──
  const defaultedBanks = [];
  if (results?.status && results?.bank_names) {
    results.status.forEach((s, i) => {
      if (s === "Default" && results.bank_names[i]) {
        defaultedBanks.push(results.bank_names[i].slice(0, 30));
      }
    });
  }

  // Game results
  const gameData = tab === "strategic" && results?.opaque ? results : null;

  return (
    <div className="relative h-screen w-screen overflow-hidden pt-16">
      {/* ── 3D GRAPH CANVAS ─────────────────────────── */}
      <div ref={containerRef} className="absolute inset-0 pt-16">
        {loading ? (
          <div className="flex h-full items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin text-stability-green" />
          </div>
        ) : topology ? (
          <NetworkGraph3D
            graphData={topology}
            statusMap={statusMap}
            width={dims.w}
            height={dims.h}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-text-muted text-sm">
            Failed to load network topology
          </div>
        )}
      </div>

      {/* ── HUD OVERLAY ─────────────────────────────── */}

      {/* Left Control Panel */}
      <div className="absolute left-4 top-20 bottom-4 w-80 z-20 flex flex-col gap-3 overflow-auto scrollbar-thin pr-1">
        {/* Tab selection */}
        <GlassPanel className="!p-3">
          <div className="flex gap-1.5 flex-wrap">
            <TabBtn
              active={tab === "mechanical"}
              onClick={() => setTab("mechanical")}
              icon={Settings}
              label="Mechanical"
            />
            <TabBtn
              active={tab === "strategic"}
              onClick={() => setTab("strategic")}
              icon={Gamepad2}
              label="Strategic"
            />
            <TabBtn
              active={tab === "climate"}
              onClick={() => setTab("climate")}
              icon={CloudLightning}
              label="Climate"
            />
          </div>
        </GlassPanel>

        {/* Controls */}
        <GlassPanel className="space-y-4 flex-1 overflow-auto">
          <div className="flex items-center gap-2 mb-2">
            <Settings className="h-4 w-4 text-text-muted" />
            <span className="text-xs font-[family-name:var(--font-mono)] uppercase tracking-wider text-text-muted">
              {tab === "mechanical"
                ? "Shock Parameters"
                : tab === "climate"
                ? "Climate Scenario"
                : "Game Theory"}
            </span>
          </div>

          <AnimatePresence mode="wait">
            {tab === "mechanical" && (
              <motion.div
                key="mech"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                className="space-y-4"
              >
                <Slider
                  label="Shock Severity"
                  value={severity}
                  onChange={setSeverity}
                  min={0}
                  max={1}
                  suffix="%"
                />
                <Slider
                  label="Intraday Steps"
                  value={nSteps}
                  onChange={(v) => setNSteps(Math.round(v))}
                  min={1}
                  max={50}
                  step={1}
                />
                <Slider
                  label="Panic Rate"
                  value={panicRate}
                  onChange={setPanicRate}
                  min={0}
                  max={0.5}
                />
                <Slider
                  label="Fire-Sale α"
                  value={fireSaleAlpha}
                  onChange={setFireSaleAlpha}
                  min={0}
                  max={0.05}
                  step={0.001}
                />
                <div className="border-t border-border pt-3">
                  <Toggle
                    label="Central Clearing (CCP)"
                    checked={useCcp}
                    onChange={setUseCcp}
                    description="Route edges through hub-and-spoke"
                  />
                  {useCcp && (
                    <div className="mt-3">
                      <Slider
                        label="Cleared Volume"
                        value={clearingRate}
                        onChange={setClearingRate}
                        min={0}
                        max={1}
                        suffix="%"
                      />
                    </div>
                  )}
                </div>
              </motion.div>
            )}

            {tab === "climate" && (
              <motion.div
                key="climate"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                className="space-y-4"
              >
                <Slider
                  label="Carbon Tax Severity"
                  value={carbonTax}
                  onChange={setCarbonTax}
                  min={0}
                  max={1}
                  suffix="%"
                />
                <Slider
                  label="Green Subsidy"
                  value={greenSubsidy}
                  onChange={setGreenSubsidy}
                  min={0}
                  max={0.5}
                  suffix="%"
                />
                <Slider
                  label="Shock Severity"
                  value={severity}
                  onChange={setSeverity}
                  min={0}
                  max={1}
                  suffix="%"
                />
                <Slider
                  label="Intraday Steps"
                  value={nSteps}
                  onChange={(v) => setNSteps(Math.round(v))}
                  min={1}
                  max={50}
                  step={1}
                />
              </motion.div>
            )}

            {tab === "strategic" && (
              <motion.div
                key="strat"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                className="space-y-4"
              >
                <Slider
                  label="True Solvency (θ)"
                  value={gameSolvency}
                  onChange={setGameSolvency}
                  min={-0.05}
                  max={0.3}
                />
                <Toggle
                  label="AI Transparency"
                  checked={gameTransparent}
                  onChange={setGameTransparent}
                  description="Accurate public signal from GNN"
                />
                <div className="glass rounded-lg p-3 text-[11px] text-text-muted">
                  <span className="text-neon-purple font-bold">Insight:</span>{" "}
                  Compares OPAQUE vs TRANSPARENT regimes side-by-side.
                  Transparent regime uses GNN risk scores as public signal to
                  prevent coordination failures.
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* EXECUTE */}
          <button
            onClick={execute}
            disabled={simulating}
            className={cn(
              "w-full flex items-center justify-center gap-2.5 rounded-xl py-3.5 text-sm font-bold transition-all",
              simulating
                ? "bg-white/5 text-text-muted cursor-not-allowed"
                : "bg-gradient-to-r from-crisis-red to-neon-purple text-white shadow-lg shadow-crisis-red/20 hover:shadow-crisis-red/40 hover:scale-[1.02] active:scale-[0.98]"
            )}
          >
            {simulating ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Running Simulation...
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                EXECUTE SIMULATION
              </>
            )}
          </button>

          {error && (
            <div className="flex items-start gap-2 rounded-lg bg-crisis-red/10 border border-crisis-red/20 p-3 text-xs text-crisis-red">
              <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
              {error}
            </div>
          )}
        </GlassPanel>
      </div>

      {/* ── Bottom Metrics Bar ──────────────────────── */}
      <AnimatePresence>
        {results && tab !== "strategic" && (
          <motion.div
            initial={{ y: 100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 100, opacity: 0 }}
            transition={{ type: "spring", damping: 25 }}
            className="absolute bottom-4 left-88 right-4 z-20"
          >
            <GlassPanel className="!p-0 overflow-hidden">
              {/* Ticker */}
              {defaultedBanks.length > 0 && (
                <div className="border-b border-border px-4 py-2">
                  <DefaultTicker defaults={defaultedBanks} />
                </div>
              )}

              <div className="p-4 flex flex-col lg:flex-row gap-4">
                {/* Metrics */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 shrink-0">
                  <MetricCard
                    label="Defaults"
                    value={results.n_defaults ?? 0}
                    prefix=""
                    suffix=""
                    color="text-crisis-red"
                  />
                  <MetricCard
                    label="Distressed"
                    value={results.n_distressed ?? 0}
                    prefix=""
                    suffix=""
                    color="text-amber-warn"
                  />
                  <MetricCard
                    label="Capital Lost"
                    value={results.equity_loss ?? 0}
                    color="text-crisis-red"
                  />
                  <MetricCard
                    label="Asset Price"
                    value={
                      results.final_asset_price
                        ? `${(results.final_asset_price * 100).toFixed(1)}%`
                        : "100%"
                    }
                    prefix=""
                    suffix=""
                    color={
                      (results.final_asset_price ?? 1) < 0.9
                        ? "text-crisis-red"
                        : "text-stability-green"
                    }
                  />
                </div>

                {/* Timeline chart */}
                {timelineData.length > 1 && (
                  <div className="flex-1 min-h-[120px]">
                    <ResponsiveContainer width="100%" height={120}>
                      <AreaChart data={timelineData}>
                        <defs>
                          <linearGradient
                            id="lossGrad"
                            x1="0"
                            y1="0"
                            x2="0"
                            y2="1"
                          >
                            <stop
                              offset="0%"
                              stopColor="#ff2a6d"
                              stopOpacity={0.3}
                            />
                            <stop
                              offset="100%"
                              stopColor="#ff2a6d"
                              stopOpacity={0}
                            />
                          </linearGradient>
                        </defs>
                        <XAxis
                          dataKey="step"
                          tick={{ fill: "#555566", fontSize: 10 }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <YAxis
                          tick={{ fill: "#555566", fontSize: 10 }}
                          axisLine={false}
                          tickLine={false}
                          width={40}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "rgba(13,13,20,0.95)",
                            border: "1px solid rgba(255,255,255,0.1)",
                            borderRadius: 8,
                            fontSize: 11,
                            fontFamily: "JetBrains Mono",
                          }}
                          labelStyle={{ color: "#888" }}
                        />
                        <Area
                          type="monotone"
                          dataKey="loss"
                          stroke="#ff2a6d"
                          fill="url(#lossGrad)"
                          strokeWidth={2}
                          name="Loss ($B)"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            </GlassPanel>
          </motion.div>
        )}

        {/* Game theory results */}
        {gameData && tab === "strategic" && (
          <motion.div
            initial={{ y: 100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 100, opacity: 0 }}
            className="absolute bottom-4 left-88 right-4 z-20"
          >
            <GlassPanel>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <p className="text-[10px] font-[family-name:var(--font-mono)] text-crisis-red uppercase tracking-wider mb-1">
                    Opaque Regime
                  </p>
                  <p className="text-2xl font-bold font-[family-name:var(--font-mono)] text-crisis-red">
                    {((gameData.opaque?.run_rate ?? 0) * 100).toFixed(1)}%
                  </p>
                  <p className="text-[10px] text-text-muted">Bank Runs</p>
                  <p className="text-sm font-[family-name:var(--font-mono)] text-crisis-red mt-1">
                    {formatUSD(gameData.opaque?.total_fire_sale_loss ?? 0)}
                  </p>
                </div>
                <div className="text-center border-x border-border px-4">
                  <p className="text-[10px] font-[family-name:var(--font-mono)] text-data-blue uppercase tracking-wider mb-1">
                    Capital Saved by AI
                  </p>
                  <p className="text-3xl font-bold font-[family-name:var(--font-mono)] text-data-blue">
                    {formatUSD(gameData.capital_saved ?? 0)}
                  </p>
                  <p className="text-[10px] text-text-muted mt-1">
                    Transparency Dividend
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-[10px] font-[family-name:var(--font-mono)] text-stability-green uppercase tracking-wider mb-1">
                    Transparent Regime
                  </p>
                  <p className="text-2xl font-bold font-[family-name:var(--font-mono)] text-stability-green">
                    {((gameData.transparent?.run_rate ?? 0) * 100).toFixed(1)}%
                  </p>
                  <p className="text-[10px] text-text-muted">Bank Runs</p>
                  <p className="text-sm font-[family-name:var(--font-mono)] text-stability-green mt-1">
                    {formatUSD(gameData.transparent?.total_fire_sale_loss ?? 0)}
                  </p>
                </div>
              </div>
            </GlassPanel>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Top-right info badge */}
      <div className="absolute top-20 right-4 z-20">
        <GlassPanel className="!p-3 flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute h-full w-full rounded-full bg-stability-green opacity-75" />
              <span className="relative rounded-full h-2 w-2 bg-stability-green" />
            </span>
            <span className="text-[10px] font-[family-name:var(--font-mono)] text-text-muted uppercase tracking-wider">
              {topology
                ? `${topology.nodes?.length ?? 0} nodes · ${topology.links?.length ?? 0} edges`
                : "Loading..."}
            </span>
          </div>
        </GlassPanel>
      </div>
    </div>
  );
}
