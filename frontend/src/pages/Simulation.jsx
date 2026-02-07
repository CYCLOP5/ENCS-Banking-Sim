import { useState, useEffect, useCallback, useRef, useMemo } from "react";
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
  Info,
  RotateCcw,
  Search,
  Square,
} from "lucide-react";
import GlassPanel from "../components/GlassPanel";
import MetricCard from "../components/MetricCard";
import NetworkGraph3D from "../components/NetworkGraph3D";
import { fetchBanks, runSimulation, runClimate, runGame } from "../services/api";
import { preloadTopology } from "../services/topologyCache";
import { cn, formatUSD } from "../lib/utils";
import DetailedAnalysis from "../components/DetailedAnalysis";
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
        <span className="text-[11px] text-text-secondary uppercase tracking-wider font-[family-name:var(--font-mono)]">
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
        <span className="text-[11px] text-text-secondary uppercase tracking-wider font-[family-name:var(--font-mono)]">
          {label}
        </span>
        {description && (
          <p className="text-[10px] text-text-muted mt-0.5">{description}</p>
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
function DefaultTicker({ defaults = [], distressed = [] }) {
  if (defaults.length === 0 && distressed.length === 0) return null;
  // Show at most 30 names to keep DOM light; full list is in detail panel
  const visibleDefaults = defaults.slice(0, 30).map(n => ({ name: n, type: 'default' }));
  const visibleDistressed = distressed.slice(0, 15).map(n => ({ name: n, type: 'distressed' }));
  const visible = [...visibleDefaults, ...visibleDistressed];
  const doubled = [...visible, ...visible];
  // Scale duration: ~2s per name, minimum 50s — slower = less CPU
  const duration = Math.max(50, visible.length * 2);
  return (
    <div className="overflow-hidden whitespace-nowrap mask-gradient">
      <div
        className="inline-flex gap-6"
        style={{
          animation: `ticker-scroll ${duration}s linear infinite`,
          willChange: "transform",
        }}
      >
        {doubled.map((item, i) => (
          <span
            key={i}
            className="inline-flex items-center gap-1.5 text-xs font-[family-name:var(--font-mono)]"
          >
            <span className={`h-1.5 w-1.5 rounded-full ${
              item.type === 'distressed' ? 'bg-orange-500' : 'bg-crisis-red'
            }`} />
            <span className={item.type === 'distressed' ? 'text-orange-500' : 'text-crisis-red'}>
              {item.name}
            </span>
            {item.type === 'distressed' && (
              <span className="text-[10px] text-orange-500/60 uppercase">distressed</span>
            )}
          </span>
        ))}
      </div>
    </div>
  );
}

/* ── Stable empty object so statusMap ref never changes when there's no status ── */
const EMPTY_STATUS_MAP = Object.freeze({});

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
  const graphRef = useRef(null);
  const [dims, setDims] = useState({ w: 800, h: 600 });
  const [detailOpen, setDetailOpen] = useState(false);
  const [liteMode, setLiteMode] = useState(true);

  // ── Contagion cascade animation ──
  const [contagionSet, setContagionSet] = useState(null);   // Set<nodeId>
  const [contagionLinks, setContagionLinks] = useState(null); // Set<"src-tgt">
  const [contagionActive, setContagionActive] = useState(false);
  const contagionTimerRef = useRef(null);

  // ── Game playback animation ──
  const [gameStep, setGameStep] = useState(null);              // current step index (0-based)
  const [gamePlaybackActive, setGamePlaybackActive] = useState(false);
  const gamePlaybackTimerRef = useRef(null);
  const [gameRegime, setGameRegime] = useState("opaque");     // "opaque" | "transparent"
  const [gameStatusMap, setGameStatusMap] = useState({});      // {topologyNodeId: "WITHDRAW"|"ROLL_OVER"}
  const [gameFlippedSet, setGameFlippedSet] = useState(null);  // Set<nodeId>

  // ── Bank list for selector ──
  const [bankList, setBankList] = useState([]);
  const [triggerIdx, setTriggerIdx] = useState(0);
  const [bankSearch, setBankSearch] = useState("");

  // ── Controls ──
  const [severity, setSeverity] = useState(1.0);
  const [nSteps, setNSteps] = useState(10);
  const [panicRate, setPanicRate] = useState(0.1);
  const [fireSaleAlpha, setFireSaleAlpha] = useState(0.005);
  const [useCcp, setUseCcp] = useState(false);
  const [clearingRate, setClearingRate] = useState(0.5);
  const [maxIter, setMaxIter] = useState(100);
  const [tolerance, setTolerance] = useState(1e-5);
  const [distressThreshold, setDistressThreshold] = useState(0.95);
  const [sigma, setSigma] = useState(0.05);
  const [marginMultiplier, setMarginMultiplier] = useState(1.0);
  const [defaultFundRatio, setDefaultFundRatio] = useState(0.05);
  const [useIntraday, setUseIntraday] = useState(true);
  // Climate
  const [carbonTax, setCarbonTax] = useState(0.5);
  const [greenSubsidy, setGreenSubsidy] = useState(0.1);
  const [climateUseIntraday, setClimateUseIntraday] = useState(true);
  // Game
  const [gameTransparent, setGameTransparent] = useState(false);
  const [gameSolvency, setGameSolvency] = useState(0.2);
  const [gameNBanks, setGameNBanks] = useState(20);
  const [gameNSteps, setGameNSteps] = useState(5);
  const [gameInterestRate, setGameInterestRate] = useState(0.10);
  const [gameRecoveryRate, setGameRecoveryRate] = useState(0.40);
  const [gameRiskAversion, setGameRiskAversion] = useState(1.0);
  const [gameNoiseStd, setGameNoiseStd] = useState(0.08);
  const [gameHaircut, setGameHaircut] = useState(0.20);
  const [gameMarginPressure, setGameMarginPressure] = useState(0.30);
  const [gameExposure, setGameExposure] = useState(1.0);

  // ── Load topology + bank list ──
  useEffect(() => {
    preloadTopology()
      .then(setTopology)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
    fetchBanks()
      .then((data) => setBankList(data.banks || []))
      .catch(() => {});
  }, []);

  // ── Resize observer (debounced to avoid re-renders on every frame) ──
  useEffect(() => {
    if (!containerRef.current) return;
    let rafId = null;
    const ro = new ResizeObserver((entries) => {
      if (rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        const { width, height } = entries[0].contentRect;
        setDims((prev) =>
          prev.w === Math.round(width) && prev.h === Math.round(height)
            ? prev
            : { w: Math.round(width), h: Math.round(height) }
        );
      });
    });
    ro.observe(containerRef.current);
    return () => { ro.disconnect(); if (rafId) cancelAnimationFrame(rafId); };
  }, []);

  // ── Build status map from results (memoized for reference stability) ──
  // Dep is results?.status — NOT results — so game results ({opaque,transparent})
  // don't produce a new {} ref, which would bust enriched's memo and crash the layout.
  const statusArray = results?.status;
  const statusMap = useMemo(() => {
    if (!statusArray) return EMPTY_STATUS_MAP;
    const map = {};
    statusArray.forEach((s, i) => { map[i] = s; });
    return map;
  }, [statusArray]);

  // ── Stop contagion animation ──
  const stopContagion = useCallback(() => {
    if (contagionTimerRef.current) {
      clearInterval(contagionTimerRef.current);
      contagionTimerRef.current = null;
    }
    setContagionActive(false);
    setContagionSet(null);
    setContagionLinks(null);
    if (graphRef.current) graphRef.current.zoomToFit(1000);
  }, []);

  // ── Stop game playback ──
  const stopGamePlayback = useCallback(() => {
    if (gamePlaybackTimerRef.current) {
      clearInterval(gamePlaybackTimerRef.current);
      gamePlaybackTimerRef.current = null;
    }
    setGamePlaybackActive(false);
    setGameStep(null);
    setGameStatusMap({});
    setGameFlippedSet(null);
    if (graphRef.current) graphRef.current.zoomToFit(1000);
  }, []);

  // ── Start game playback ──
  const startGamePlayback = useCallback(() => {
    if (!topology || !results?.opaque) return;
    // Clear any previous
    if (gamePlaybackTimerRef.current) clearInterval(gamePlaybackTimerRef.current);
    stopContagion(); // ensure contagion is off

    const regime = results[gameRegime] ?? results.opaque;
    const timeline = regime?.timeline;
    if (!timeline?.decisions?.length) return;

    const nStepsGame = timeline.decisions.length;
    const nAgents = timeline.decisions[0]?.length ?? 0;
    const topoNodes = topology.nodes;
    const mappableCount = Math.min(nAgents, topoNodes.length);

    // Stop physics for clean visualization
    if (graphRef.current) graphRef.current.stopPhysics();

    let stepIdx = 0;
    let prevDecisions = null;

    // Initial step
    const buildStepState = (sIdx) => {
      const decisions = timeline.decisions[sIdx];
      const statusObj = {};
      const flipped = new Set();

      for (let i = 0; i < mappableCount; i++) {
        const nodeId = topoNodes[i].id;
        statusObj[nodeId] = decisions[i]; // "WITHDRAW" or "ROLL_OVER"
        if (prevDecisions && prevDecisions[i] !== decisions[i]) {
          flipped.add(nodeId);
        }
      }
      prevDecisions = [...decisions];
      return { statusObj, flipped };
    };

    setGamePlaybackActive(true);
    // Show first step immediately
    const { statusObj, flipped } = buildStepState(0);
    setGameStep(0);
    setGameStatusMap(statusObj);
    setGameFlippedSet(flipped);

    // Zoom to the subgraph of game agents
    if (graphRef.current && mappableCount > 0) {
      graphRef.current.focusNode(topoNodes[0].id, 250);
    }

    stepIdx = 1;
    gamePlaybackTimerRef.current = setInterval(() => {
      if (stepIdx >= nStepsGame) {
        clearInterval(gamePlaybackTimerRef.current);
        gamePlaybackTimerRef.current = null;
        // Zoom out after finishing
        if (graphRef.current) graphRef.current.zoomToFit(2000);
        setTimeout(() => {
          setGamePlaybackActive(false);
          setGameFlippedSet(null);
        }, 3000);
        return;
      }

      const { statusObj: sObj, flipped: fl } = buildStepState(stepIdx);
      setGameStep(stepIdx);
      setGameStatusMap(sObj);
      setGameFlippedSet(fl);

      // Camera: follow a withdrawing node for dramatic effect
      if (graphRef.current && stepIdx < 10) {
        const withdrawIds = Object.entries(sObj)
          .filter(([, d]) => d === "WITHDRAW")
          .map(([id]) => id);
        if (withdrawIds.length > 0) {
          graphRef.current.focusNode(withdrawIds[0], 200 + stepIdx * 20);
        }
      } else if (graphRef.current && stepIdx === 10) {
        graphRef.current.zoomToFit(1500);
      }

      stepIdx++;
    }, 1200); // 1.2s per step
  }, [topology, results, gameRegime, stopContagion]);

  // Auto-start game playback when strategic results arrive
  useEffect(() => {
    if (tab === "strategic" && results?.opaque) startGamePlayback();
    return () => {
      if (gamePlaybackTimerRef.current) clearInterval(gamePlaybackTimerRef.current);
    };
  }, [results, tab]); // eslint-disable-line react-hooks/exhaustive-deps

  // Restart playback when regime toggle changes
  useEffect(() => {
    if (tab === "strategic" && results?.opaque && !simulating) {
      startGamePlayback();
    }
  }, [gameRegime]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── BFS contagion animation ──
  const startContagion = useCallback(() => {
    if (!topology || !results?.status) return;
    // Clear previous animation
    if (contagionTimerRef.current) clearInterval(contagionTimerRef.current);

    // Build adjacency from topology links (with edge info)
    const adj = new Map();       // nodeId -> [{neighbor, src, tgt}]
    for (const l of topology.links) {
      const src = l.source?.id ?? l.source;
      const tgt = l.target?.id ?? l.target;
      if (!adj.has(src)) adj.set(src, []);
      if (!adj.has(tgt)) adj.set(tgt, []);
      adj.get(src).push({ neighbor: tgt, key: `${src}-${tgt}` });
      adj.get(tgt).push({ neighbor: src, key: `${tgt}-${src}` });
    }

    // Collect all defaulted/distressed node ids
    const affected = new Set();
    results.status.forEach((s, i) => {
      if (s === "Default" || s === "Distressed") affected.add(i);
    });
    if (affected.size === 0) return;

    // Find the trigger bank (index 0 from API, or first Default)
    const trigger = results.trigger_idx ?? [...affected][0] ?? 0;

    // BFS from trigger through ALL nodes (not just affected) to build reachable graph
    // This ensures we trace the transaction path even through safe intermediaries
    const bfsOrder = [];     // array of { nodeId, parentId, linkKey } in BFS order
    const visited = new Set();
    let frontier = [{ nodeId: trigger, parentId: null, linkKey: null }];
    visited.add(trigger);
    bfsOrder.push({ nodeId: trigger, parentId: null, linkKey: null });

    while (frontier.length > 0) {
      const next = [];
      for (const { nodeId } of frontier) {
        for (const { neighbor, key } of (adj.get(nodeId) || [])) {
          if (!visited.has(neighbor)) {
            visited.add(neighbor);
            const entry = { nodeId: neighbor, parentId: nodeId, linkKey: key };
            bfsOrder.push(entry);
            // Only expand further through affected nodes to trace contagion paths
            if (affected.has(neighbor)) {
              next.push(entry);
            }
          }
        }
      }
      frontier = next;
    }

    // Add any affected nodes unreachable from trigger
    const unreached = [...affected].filter((id) => !visited.has(id));
    for (const id of unreached) {
      bfsOrder.push({ nodeId: id, parentId: null, linkKey: null });
    }

    // Build animation steps: reveal 1-3 nodes per step for slow cascade
    // Prioritize affected nodes; batch safe intermediaries together
    const steps = []; // each step: { nodes: [id], links: [key] }
    let currentStep = { nodes: [], links: [] };
    let safeBuffer = { nodes: [], links: [] };

    for (const entry of bfsOrder) {
      if (affected.has(entry.nodeId)) {
        // Flush any safe buffer first (as background context)
        if (safeBuffer.nodes.length > 0) {
          steps.push({ ...safeBuffer });
          safeBuffer = { nodes: [], links: [] };
        }
        // Each affected node gets its own step for dramatic reveal
        currentStep = { nodes: [entry.nodeId], links: entry.linkKey ? [entry.linkKey] : [] };
        steps.push(currentStep);
      } else {
        // Safe nodes: batch into groups of 5
        safeBuffer.nodes.push(entry.nodeId);
        if (entry.linkKey) safeBuffer.links.push(entry.linkKey);
        if (safeBuffer.nodes.length >= 5) {
          steps.push({ ...safeBuffer });
          safeBuffer = { nodes: [], links: [] };
        }
      }
    }
    if (safeBuffer.nodes.length > 0) steps.push(safeBuffer);

    // Cap total steps to 50 (merge late steps)
    const MAX_STEPS = 50;
    if (steps.length > MAX_STEPS) {
      const merged = { nodes: [], links: [] };
      for (let i = MAX_STEPS - 1; i < steps.length; i++) {
        merged.nodes.push(...steps[i].nodes);
        merged.links.push(...steps[i].links);
      }
      steps.length = MAX_STEPS - 1;
      steps.push(merged);
    }

    // Animate step by step
    let stepIdx = 0;
    const revealedNodes = new Set();
    const revealedLinks = new Set();
    setContagionActive(true);
    setContagionSet(new Set());
    setContagionLinks(new Set());

    // FIX: Stop physics so nodes stop moving while we zoom
    if (graphRef.current) {
      graphRef.current.stopPhysics();
    }

    // Zoom into trigger node first
    if (graphRef.current) {
      graphRef.current.focusNode(trigger, 180);
    }

    contagionTimerRef.current = setInterval(() => {
      if (stepIdx >= steps.length) {
        clearInterval(contagionTimerRef.current);
        contagionTimerRef.current = null;
        // Zoom out to see everything
        if (graphRef.current) graphRef.current.zoomToFit(2000);
        setTimeout(() => {
          setContagionActive(false);
          setContagionLinks(null);
        }, 3000);
        return;
      }

      const step = steps[stepIdx];
      for (const nid of step.nodes) revealedNodes.add(nid);
      for (const lk of step.links) revealedLinks.add(lk);
      setContagionSet(new Set(revealedNodes));
      setContagionLinks(new Set(revealedLinks));

      // Camera follows the cascade for the first 10 affected-node steps
      const affectedInStep = step.nodes.filter((n) => affected.has(n));
      if (graphRef.current && affectedInStep.length > 0 && stepIdx < 15) {
        const dist = 180 + stepIdx * 25; // gradually pull back from farther away
        graphRef.current.focusNode(affectedInStep[0], dist);
      } else if (graphRef.current && stepIdx === 15) {
        graphRef.current.zoomToFit(1500);
      }

      stepIdx++;
    }, 1400); // 1.4 seconds per step — slow, dramatic cascade
  }, [topology, results]);

  // Auto-start contagion on new results
  useEffect(() => {
    if (results?.status) startContagion();
    return () => {
      if (contagionTimerRef.current) clearInterval(contagionTimerRef.current);
    };
  }, [results]); // eslint-disable-line react-hooks/exhaustive-deps

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
          useIntraday: climateUseIntraday,
          triggerIdx,
          severity,
          nSteps,
        });
      } else if (tab === "strategic") {
        res = await runGame({
          trueSolvency: gameSolvency,
          nBanks: gameNBanks,
          nSteps: gameNSteps,
          interestRate: gameInterestRate,
          recoveryRate: gameRecoveryRate,
          riskAversion: gameRiskAversion,
          noiseStd: gameNoiseStd,
          haircut: gameHaircut,
          marginPressure: gameMarginPressure,
          exposure: gameExposure * 1e9,
        });
      } else {
        res = await runSimulation({
          triggerIdx,
          severity,
          maxIter,
          tolerance,
          distressThreshold,
          useIntraday,
          nSteps,
          sigma,
          panicRate,
          fireSaleAlpha,
          marginMultiplier,
          useCcp,
          clearingRate,
          defaultFundRatio,
        });
      }
      setResults(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setSimulating(false);
    }
  }, [
    tab, triggerIdx, severity, nSteps, panicRate, fireSaleAlpha, useCcp, clearingRate,
    maxIter, tolerance, distressThreshold, sigma, marginMultiplier, defaultFundRatio, useIntraday,
    carbonTax, greenSubsidy, climateUseIntraday,
    gameSolvency, gameNBanks, gameNSteps, gameInterestRate, gameRecoveryRate,
    gameRiskAversion, gameNoiseStd, gameHaircut, gameMarginPressure, gameExposure,
  ]);

  // ── Timeline chart data ──
  const timelineData =
    results?.equity_loss_timeline?.map((val, i) => ({
      step: i,
      loss: val / 1e9,
      defaults: results.defaults_timeline?.[i] ?? 0,
      price: results.price_timeline?.[i] ?? 1,
    })) ?? [];

  // ── Defaulted & distressed bank names ──
  const defaultedBanks = [];
  const distressedBanks = [];
  if (results?.status && results?.bank_names) {
    results.status.forEach((s, i) => {
      if (s === "Default" && results.bank_names[i]) {
        defaultedBanks.push(results.bank_names[i].slice(0, 30));
      } else if (s === "Distressed" && results.bank_names[i]) {
        distressedBanks.push(results.bank_names[i].slice(0, 30));
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
            ref={graphRef}
            graphData={topology}
            statusMap={statusMap}
            contagionSet={contagionSet}
            contagionLinks={contagionLinks}
            contagionActive={contagionActive}
            gameStatusMap={gameStatusMap}
            gameActive={gamePlaybackActive}
            gameFlippedSet={gameFlippedSet}
            width={dims.w}
            height={dims.h}
            maxNodes={liteMode ? 500 : Infinity}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-text-muted text-sm">
            Failed to load network topology
          </div>
        )}
      </div>

      {/* ── HUD OVERLAY ─────────────────────────────── */}

      {/* Left Control Panel */}
      <div data-lenis-prevent className="absolute left-4 top-20 bottom-4 w-80 z-20 flex flex-col gap-3 overflow-auto scrollbar-thin pr-1">
        {/* Tab selection */}
        <GlassPanel className="!p-3">
          <div className="flex gap-1.5 flex-wrap">
            <TabBtn
              active={tab === "mechanical"}
              onClick={() => { if (tab !== "mechanical") { stopContagion(); stopGamePlayback(); setResults(null); } setTab("mechanical"); }}
              icon={Settings}
              label="Mechanical"
            />
            <TabBtn
              active={tab === "strategic"}
              onClick={() => { if (tab !== "strategic") { stopContagion(); stopGamePlayback(); setResults(null); } setTab("strategic"); }}
              icon={Gamepad2}
              label="Strategic"
            />
            <TabBtn
              active={tab === "climate"}
              onClick={() => { if (tab !== "climate") { stopContagion(); stopGamePlayback(); setResults(null); } setTab("climate"); }}
              icon={CloudLightning}
              label="Climate"
            />
          </div>
        </GlassPanel>

        {/* Controls */}
        <GlassPanel className="space-y-4 flex-1 overflow-auto">
          <div className="flex items-center gap-2 mb-2">
            <Settings className="h-4 w-4 text-text-secondary" />
            <span className="text-xs font-[family-name:var(--font-mono)] uppercase tracking-wider text-text-secondary">
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
                {/* Trigger Bank Selector */}
                <div className="space-y-1.5">
                  <span className="text-[11px] text-text-secondary uppercase tracking-wider font-[family-name:var(--font-mono)]">
                    Trigger Bank
                  </span>
                  <div className="relative">
                    <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-text-muted pointer-events-none" />
                    <input
                      type="text"
                      value={bankSearch}
                      onChange={(e) => setBankSearch(e.target.value)}
                      placeholder="Search banks..."
                      className="w-full pl-8 pr-3 py-2 rounded-lg bg-white/[0.04] border border-border text-xs text-text-primary
                        font-[family-name:var(--font-mono)] placeholder:text-text-muted/60
                        focus:outline-none focus:border-stability-green/40 transition-colors"
                    />
                  </div>
                  {bankList.length > 0 && (
                    <div className="max-h-32 overflow-auto rounded-lg border border-border bg-void-panel">
                      {bankList
                        .filter((b) =>
                          bankSearch
                            ? b.name.toLowerCase().includes(bankSearch.toLowerCase())
                            : true
                        )
                        .slice(0, 50)
                        .map((b) => (
                          <button
                            key={b.id}
                            onClick={() => {
                              setTriggerIdx(b.id);
                              setBankSearch(b.name);
                            }}
                            className={cn(
                              "w-full text-left px-3 py-1.5 text-[11px] font-[family-name:var(--font-mono)] transition-colors truncate",
                              b.id === triggerIdx
                                ? "bg-crisis-red/15 text-crisis-red"
                                : "text-text-secondary hover:bg-white/[0.04] hover:text-text-primary"
                            )}
                          >
                            {b.name.slice(0, 40)}
                            <span className="ml-1 text-text-muted">
                              · ${(b.total_assets / 1e9).toFixed(0)}B
                            </span>
                          </button>
                        ))}
                    </div>
                  )}
                </div>

                <Slider
                  label="Shock Severity"
                  value={severity}
                  onChange={setSeverity}
                  min={0}
                  max={1}
                  suffix="%"
                />

                <div className="border-t border-border pt-3">
                  <Toggle
                    label="Intraday Engine"
                    checked={useIntraday}
                    onChange={setUseIntraday}
                    description="Run multi-step intraday simulation"
                  />
                </div>

                {useIntraday && (
                  <>
                    <Slider
                      label="Intraday Steps"
                      value={nSteps}
                      onChange={(v) => setNSteps(Math.round(v))}
                      min={1}
                      max={50}
                      step={1}
                    />
                    <Slider
                      label="Market Uncertainty (σ)"
                      value={sigma}
                      onChange={setSigma}
                      min={0.01}
                      max={0.30}
                      step={0.01}
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
                    <Slider
                      label="Margin Sensitivity"
                      value={marginMultiplier}
                      onChange={setMarginMultiplier}
                      min={0}
                      max={5}
                      step={0.1}
                    />
                  </>
                )}

                <div className="border-t border-border pt-3 space-y-3">
                  <span className="text-[10px] text-text-muted uppercase tracking-wider font-[family-name:var(--font-mono)]">
                    Convergence
                  </span>
                  <Slider
                    label="Distress Threshold"
                    value={distressThreshold}
                    onChange={setDistressThreshold}
                    min={0}
                    max={1}
                  />
                  <Slider
                    label="Max Iterations"
                    value={maxIter}
                    onChange={(v) => setMaxIter(Math.round(v))}
                    min={10}
                    max={500}
                    step={10}
                  />
                </div>

                <div className="border-t border-border pt-3">
                  <Toggle
                    label="Central Clearing (CCP)"
                    checked={useCcp}
                    onChange={setUseCcp}
                    description="Route edges through hub-and-spoke"
                  />
                  {useCcp && (
                    <div className="mt-3 space-y-3">
                      <Slider
                        label="Cleared Volume"
                        value={clearingRate}
                        onChange={setClearingRate}
                        min={0}
                        max={1}
                        suffix="%"
                      />
                      <Slider
                        label="Default Fund %"
                        value={defaultFundRatio}
                        onChange={setDefaultFundRatio}
                        min={0.01}
                        max={0.25}
                        step={0.01}
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

                <div className="border-t border-border pt-3">
                  <Toggle
                    label="Intraday Engine"
                    checked={climateUseIntraday}
                    onChange={setClimateUseIntraday}
                    description="Multi-step cascade after transition shock"
                  />
                </div>

                {climateUseIntraday && (
                  <>
                    <Slider
                      label="Intraday Steps"
                      value={nSteps}
                      onChange={(v) => setNSteps(Math.round(v))}
                      min={1}
                      max={50}
                      step={1}
                    />
                    <Slider
                      label="Shock Severity"
                      value={severity}
                      onChange={setSeverity}
                      min={0}
                      max={1}
                      suffix="%"
                    />
                  </>
                )}

                <div className="glass rounded-lg p-3 text-[11px] text-text-muted">
                  <span className="text-stability-green font-bold">Green Swan:</span>{" "}
                  Carbon tax reprices brown assets, subsidy buffers green-aligned banks.
                  Intraday engine cascades fire-sale spirals through the network.
                </div>
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
                <Slider
                  label="Number of Agents"
                  value={gameNBanks}
                  onChange={(v) => setGameNBanks(Math.round(v))}
                  min={5}
                  max={100}
                  step={1}
                />
                <Slider
                  label="Time Steps"
                  value={gameNSteps}
                  onChange={(v) => setGameNSteps(Math.round(v))}
                  min={2}
                  max={20}
                  step={1}
                />

                <div className="border-t border-border pt-3 space-y-3">
                  <span className="text-[10px] text-text-muted uppercase tracking-wider font-[family-name:var(--font-mono)]">
                    Market Parameters
                  </span>
                  <Slider
                    label="Interest Rate (r)"
                    value={gameInterestRate}
                    onChange={setGameInterestRate}
                    min={0.01}
                    max={0.20}
                    step={0.01}
                  />
                  <Slider
                    label="Recovery Rate (R)"
                    value={gameRecoveryRate}
                    onChange={setGameRecoveryRate}
                    min={0.10}
                    max={0.80}
                    step={0.01}
                  />
                  <Slider
                    label="Exposure / Bank ($B)"
                    value={gameExposure}
                    onChange={setGameExposure}
                    min={0.1}
                    max={50}
                    step={0.1}
                    suffix="B"
                  />
                </div>

                <div className="border-t border-border pt-3 space-y-3">
                  <span className="text-[10px] text-text-muted uppercase tracking-wider font-[family-name:var(--font-mono)]">
                    Agent Behavior
                  </span>
                  <Slider
                    label="Risk Aversion (λ)"
                    value={gameRiskAversion}
                    onChange={setGameRiskAversion}
                    min={0.1}
                    max={3.0}
                    step={0.1}
                  />
                  <Slider
                    label="Private Noise (σ)"
                    value={gameNoiseStd}
                    onChange={setGameNoiseStd}
                    min={0.01}
                    max={0.30}
                    step={0.01}
                  />
                  <Slider
                    label="Fire-Sale Haircut"
                    value={gameHaircut}
                    onChange={setGameHaircut}
                    min={0.05}
                    max={0.50}
                    step={0.01}
                  />
                  <Slider
                    label="Margin Volatility"
                    value={gameMarginPressure}
                    onChange={setGameMarginPressure}
                    min={0}
                    max={1}
                    step={0.01}
                  />
                </div>

                <div className="border-t border-border pt-3">
                  <Toggle
                    label="AI Transparency"
                    checked={gameTransparent}
                    onChange={setGameTransparent}
                    description="Accurate public signal from GNN"
                  />
                </div>

                <div className="glass rounded-lg p-3 text-[11px] text-text-muted">
                  <span className="text-neon-purple font-bold">Insight:</span>{" "}
                  Compares OPAQUE vs TRANSPARENT regimes side-by-side.
                  Transparent regime uses GNN risk-frequency scores as public signal to
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
              {(defaultedBanks.length > 0 || distressedBanks.length > 0) && (
                <div className="border-b border-border px-4 py-2">
                  <DefaultTicker defaults={defaultedBanks} distressed={distressedBanks} />
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
            <GlassPanel className="!p-0 overflow-hidden">
              {/* Step progress bar during playback */}
              {gamePlaybackActive && gameStep !== null && (() => {
                const regime = gameData[gameRegime] ?? gameData.opaque;
                const tl = regime?.timeline;
                const totalSteps = tl?.decisions?.length ?? 1;
                const nWithdrawals = tl?.n_runs?.[gameStep] ?? 0;
                const stepLoss = tl?.step_fire_sale_loss?.[gameStep] ?? 0;
                return (
                  <div className="border-b border-border px-4 py-2 flex items-center gap-4">
                    <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-neon-purple to-crisis-red transition-all duration-500"
                        style={{ width: `${((gameStep + 1) / totalSteps) * 100}%` }}
                      />
                    </div>
                    <span className="text-[11px] font-[family-name:var(--font-mono)] text-text-secondary whitespace-nowrap">
                      Step {gameStep + 1}/{totalSteps}
                      <span className="mx-2 text-text-muted">·</span>
                      <span className="text-crisis-red">{nWithdrawals} withdrew</span>
                      <span className="mx-2 text-text-muted">·</span>
                      <span className="text-crisis-red">${(stepLoss / 1e9).toFixed(2)}B loss</span>
                    </span>
                  </div>
                );
              })()}

              <div className="p-4">
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
              </div>
            </GlassPanel>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Top-right info badge + detail button */}
      <div className="absolute top-20 right-4 z-20 flex flex-col gap-2 items-end">
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
          <button
            onClick={() => setLiteMode((v) => !v)}
            className={cn(
              "ml-2 px-2.5 py-1 rounded-lg text-[10px] font-bold font-[family-name:var(--font-mono)] uppercase tracking-wider transition-all border",
              liteMode
                ? "bg-stability-green/20 text-stability-green border-stability-green/40"
                : "bg-white/5 text-text-muted border-border hover:border-text-muted"
            )}
          >
            {liteMode ? "LITE ✓" : "LITE"}
          </button>
        </GlassPanel>

        {/* Detailed Analysis Button */}
        {results && (
          <motion.button
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            onClick={() => {
              setDetailOpen(true);
              if (graphRef.current) graphRef.current.pauseRendering();
            }}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl glass-bright border border-border-bright
              hover:border-stability-green/40 hover:shadow-lg hover:shadow-stability-green/10
              transition-all group cursor-pointer"
          >
            <Info className="h-4 w-4 text-stability-green group-hover:scale-110 transition-transform" />
            <span className="text-xs font-[family-name:var(--font-mono)] font-semibold text-text-primary group-hover:text-stability-green transition-colors">
              DETAILED ANALYSIS
            </span>
          </motion.button>
        )}

        {/* Replay/Stop Contagion Button */}
        {results && results.status && (
          <motion.button
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            onClick={contagionActive ? stopContagion : startContagion}
            className={cn(
              "flex items-center gap-2 px-4 py-2.5 rounded-xl glass-bright border transition-all group cursor-pointer",
              contagionActive
                ? "border-crisis-red/40 shadow-lg shadow-crisis-red/10 bg-crisis-red/10"
                : "border-border-bright hover:border-crisis-red/40 hover:shadow-lg hover:shadow-crisis-red/10"
            )}
          >
            {contagionActive ? (
              <Square className="h-4 w-4 text-crisis-red fill-crisis-red" />
            ) : (
              <RotateCcw className="h-4 w-4 text-crisis-red group-hover:scale-110 transition-transform" />
            )}
            <span className="text-xs font-[family-name:var(--font-mono)] font-semibold text-text-primary group-hover:text-crisis-red transition-colors">
              {contagionActive ? "STOP REPLAY" : "REPLAY CONTAGION"}
            </span>
          </motion.button>
        )}

        {/* Game Replay/Stop Button */}
        {results && results.opaque && tab === "strategic" && (
          <motion.button
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            onClick={gamePlaybackActive ? stopGamePlayback : startGamePlayback}
            className={cn(
              "flex items-center gap-2 px-4 py-2.5 rounded-xl glass-bright border transition-all group cursor-pointer",
              gamePlaybackActive
                ? "border-neon-purple/40 shadow-lg shadow-neon-purple/10 bg-neon-purple/10"
                : "border-border-bright hover:border-neon-purple/40 hover:shadow-lg hover:shadow-neon-purple/10"
            )}
          >
            {gamePlaybackActive ? (
              <Square className="h-4 w-4 text-neon-purple fill-neon-purple" />
            ) : (
              <RotateCcw className="h-4 w-4 text-neon-purple group-hover:scale-110 transition-transform" />
            )}
            <span className="text-xs font-[family-name:var(--font-mono)] font-semibold text-text-primary group-hover:text-neon-purple transition-colors">
              {gamePlaybackActive ? "STOP GAME" : "REPLAY GAME"}
            </span>
          </motion.button>
        )}

        {/* Regime toggle for game playback */}
        {results && results.opaque && tab === "strategic" && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-1 p-1 rounded-lg glass-bright border border-border-bright"
          >
            <button
              onClick={() => setGameRegime("opaque")}
              className={cn(
                "px-3 py-1.5 rounded-md text-[10px] font-bold font-[family-name:var(--font-mono)] uppercase tracking-wider transition-all",
                gameRegime === "opaque"
                  ? "bg-crisis-red/20 text-crisis-red border border-crisis-red/40"
                  : "text-text-muted hover:text-text-secondary"
              )}
            >
              OPAQUE
            </button>
            <button
              onClick={() => setGameRegime("transparent")}
              className={cn(
                "px-3 py-1.5 rounded-md text-[10px] font-bold font-[family-name:var(--font-mono)] uppercase tracking-wider transition-all",
                gameRegime === "transparent"
                  ? "bg-stability-green/20 text-stability-green border border-stability-green/40"
                  : "text-text-muted hover:text-text-secondary"
              )}
            >
              TRANSPARENT
            </button>
          </motion.div>
        )}
      </div>

      {/* ── Detailed Analysis Modal ── */}
      <DetailedAnalysis
        open={detailOpen}
        onClose={() => {
          setDetailOpen(false);
          if (graphRef.current) graphRef.current.resumeRendering();
        }}
        results={results}
        tab={tab}
      />
    </div>
  );
}
