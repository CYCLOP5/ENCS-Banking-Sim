import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  TrendingDown,
  AlertTriangle,
  BarChart3,
  Shield,
  Flame,
  DollarSign,
  Globe,
  ChevronDown,
  ChevronUp,
  Users,
  Square,
} from "lucide-react";
import GlassPanel from "./GlassPanel";
import { cn, formatUSD } from "../lib/utils";
import { explainRun } from "../services/api";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  ComposedChart,
} from "recharts";

/* ── Shared chart tooltip style ─────────────────────────────── */
const tooltipStyle = {
  background: "rgba(13,13,20,0.95)",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 8,
  fontSize: 11,
  fontFamily: "JetBrains Mono",
};

/* ── Mini stat card inside the modal ─────────────────────────── */
function StatCard({ label, value, icon: Icon, color = "text-stability-green" }) {
  return (
    <div className="glass rounded-xl p-4 flex flex-col gap-1">
      <div className="flex items-center gap-2">
        {Icon && <Icon className={cn("h-3.5 w-3.5", color)} />}
        <span className="text-[10px] font-medium uppercase tracking-widest text-text-muted font-[family-name:var(--font-mono)]">
          {label}
        </span>
      </div>
      <span className={cn("text-xl font-bold tracking-tight font-[family-name:var(--font-mono)]", color)}>
        {value}
      </span>
    </div>
  );
}

/* ── Collapsible Section ────────────────────────────────────── */
function Section({ title, icon: Icon, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-border rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-3 bg-surface hover:bg-surface-hover transition-colors"
      >
        <div className="flex items-center gap-2">
          {Icon && <Icon className="h-4 w-4 text-text-muted" />}
          <span className="text-sm font-semibold text-text-primary font-[family-name:var(--font-display)]">
            {title}
          </span>
        </div>
        {open ? (
          <ChevronUp className="h-4 w-4 text-text-muted" />
        ) : (
          <ChevronDown className="h-4 w-4 text-text-muted" />
        )}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="p-5 space-y-4">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   MECHANICAL / CLIMATE DETAIL PANEL
   ════════════════════════════════════════════════════════════════ */
function MechanicalDetail({ results }) {
  if (!results) return null;

  const status = results.status ?? [];
  const bankNames = results.bank_names ?? [];
  const isValidName = (name) => {
    if (name === null || name === undefined) return false;
    const s = String(name).trim();
    if (s.length === 0) return false;
    const lower = s.toLowerCase();
    return lower !== "nan" && lower !== "0";
  };

  const isValidEntry = (entry) => {
    if (!isValidName(entry.name)) return false;
    const nameStr = String(entry.name).trim();
    const isPlaceholder = /^bank\s+\d+$/i.test(nameStr);
    const allZero = (entry.initial ?? 0) === 0 && (entry.final ?? 0) === 0;
    return !(isPlaceholder && allZero);
  };
  const initialEq = results.initial_equity ?? [];
  const finalEq = results.final_equity ?? [];

  // ── Top casualties ──
  const casualties = bankNames
    .map((name, i) => ({
      name: (typeof name === "string" ? name.slice(0, 35) : String(name ?? `Bank ${i}`)).slice(0, 35),
      initial: initialEq[i] ?? 0,
      final: finalEq[i] ?? 0,
      loss: (initialEq[i] ?? 0) - (finalEq[i] ?? 0),
      status: status[i] ?? "Safe",
    }))
    .filter((c) => isValidEntry(c))
    .sort((a, b) => b.loss - a.loss)
    .slice(0, 15);

  // ── Timeline data ──
  const priceTl = results.price_timeline ?? [];
  const defaultsTl = results.defaults_timeline ?? [];
  const distressedTl = results.distressed_timeline ?? [];
  const gridlockTl = results.gridlock_timeline ?? [];
  const marginTl = results.margin_calls_timeline ?? [];
  const equityLossTl = results.equity_loss_timeline ?? [];

  const timelineData = priceTl.map((_, i) => ({
    step: i + 1,
    price: priceTl[i] ?? 1,
    defaults: defaultsTl[i] ?? 0,
    distressed: distressedTl[i] ?? 0,
    gridlock: gridlockTl[i] ?? 0,
    marginCalls: (marginTl[i] ?? 0) / 1e9,
    equityLoss: (equityLossTl[i] ?? 0) / 1e9,
  }));

  // ── Regional breakdown ──
  const regionCounts = {};
  bankNames.forEach((name, i) => {
    // Infer region from bank index (approximate: check results for region info)
    const st = status[i] ?? "Safe";
    if (!regionCounts[st]) regionCounts[st] = 0;
    regionCounts[st]++;
  });

  const statusPie = [
    { name: "Safe", value: status.filter((s) => s === "Safe").length, fill: "#05d5fa" },
    { name: "Distressed", value: status.filter((s) => s === "Distressed").length, fill: "#ffaa00" },
    { name: "Default", value: status.filter((s) => s === "Default").length, fill: "#ff2a6d" },
  ].filter((d) => d.name === "Safe" && d.value === 1 ? false : d.value > 0);

  const totalRelevant = statusPie.reduce((acc, curr) => acc + curr.value, 0);
  const safeTotal = totalRelevant > 0 ? totalRelevant : 1;

  // ── Bank lists by status ──
  const defaultBanks = [];
  const distressedBanks = [];
  const safeBanks = [];
  bankNames.forEach((name, i) => {
    const st = status[i] ?? "Safe";
    const entry = {
      name: (typeof name === "string" ? name.slice(0, 40) : String(name ?? `Bank ${i}`)).slice(0, 40),
      initial: initialEq[i] ?? 0,
      final: finalEq[i] ?? 0,
      loss: (initialEq[i] ?? 0) - (finalEq[i] ?? 0),
    };
    if (!isValidEntry(entry)) return;
    if (st === "Default") defaultBanks.push(entry);
    else if (st === "Distressed") distressedBanks.push(entry);
    else safeBanks.push(entry);
  });
  defaultBanks.sort((a, b) => b.loss - a.loss);
  distressedBanks.sort((a, b) => b.loss - a.loss);
  safeBanks.sort((a, b) => (b.final - a.final));

  return (
    <>
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        <StatCard label="Defaults" value={results.n_defaults ?? 0} icon={AlertTriangle} color="text-crisis-red" />
        <StatCard label="Distressed" value={results.n_distressed ?? 0} icon={Flame} color="text-amber-warn" />
        <StatCard
          label="Capital Lost"
          value={formatUSD(results.equity_loss ?? 0)}
          icon={DollarSign}
          color="text-crisis-red"
        />
        <StatCard
          label="Asset Price"
          value={
            results.final_asset_price !== undefined
              ? (results.final_asset_price * 100 < 0.1 && results.final_asset_price > 0
                ? "< 0.1%"
                : `${(results.final_asset_price * 100).toFixed(1)}%`)
              : "100%"
          }
          icon={TrendingDown}
          color={
            (results.final_asset_price ?? 1) < 0.9 ? "text-crisis-red" : "text-stability-green"
          }
        />
        <StatCard
          label="Trigger Bank"
          value={(typeof results.trigger_name === "string" ? results.trigger_name : String(results.trigger_name ?? "Bank 0")).slice(0, 20)}
          icon={Shield}
          color="text-neon-purple"
        />
        <StatCard
          label="Shock"
          value={`${((results.loss_severity ?? 1) * 100).toFixed(0)}%`}
          icon={BarChart3}
          color="text-text-secondary"
        />
      </div>

      {/* Circuit Breaker Alert */}
      {results.circuit_breaker_triggered && (
        <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-amber-500/10 border border-amber-500/30">
          <Square className="h-4 w-4 text-amber-400" />
          <span className="text-xs font-[family-name:var(--font-mono)] text-amber-400 font-bold uppercase tracking-wider">
            Circuit Breaker Triggered
          </span>
          <span className="text-[10px] text-amber-400/70 font-[family-name:var(--font-mono)]">
            — trading halted at step {results.circuit_breaker_step}, preventing further cascade
          </span>
        </div>
      )}

      {/* ── Status Distribution ── */}
      <Section title="Bank Status Distribution" icon={BarChart3}>
        <div className="space-y-4">
          {/* Stat cards row */}
          <div className="grid grid-cols-3 gap-3">
            {statusPie.length === 0 ? (
              <div className="col-span-3 text-center text-xs text-text-muted">
                No status data available.
              </div>
            ) : (
              statusPie.map((s) => (
                <div key={s.name} className="glass rounded-xl p-4 text-center">
                  <div className="text-3xl font-bold font-[family-name:var(--font-mono)]" style={{ color: s.fill }}>
                    {s.value.toLocaleString()}
                  </div>
                  <div className="text-[10px] text-text-secondary uppercase tracking-widest mt-1 font-[family-name:var(--font-mono)]">
                    {s.name}
                  </div>
                  <div className="text-xs text-text-muted font-[family-name:var(--font-mono)]">
                    {((s.value / safeTotal) * 100).toFixed(1)}%
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Stacked progress bar */}
          <div className="space-y-2">
            <div className="flex h-4 w-full rounded-full overflow-hidden bg-white/5">
              {statusPie.map((s) => {
                const pct = (s.value / safeTotal) * 100;
                if (pct === 0) return null;
                return (
                  <div
                    key={s.name}
                    style={{ width: `${pct}%`, backgroundColor: s.fill }}
                    className="h-full transition-all duration-500 first:rounded-l-full last:rounded-r-full"
                    title={`${s.name}: ${s.value} (${pct.toFixed(1)}%)`}
                  />
                );
              })}
            </div>
            <div className="flex justify-between text-[10px] text-text-muted font-[family-name:var(--font-mono)]">
              {statusPie.map((s) => (
                <div key={s.name} className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full" style={{ backgroundColor: s.fill }} />
                  <span>{s.name}</span>
                </div>
              ))}
              <span className="text-text-secondary">
                Total: {totalRelevant.toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      </Section>

      {/* ── Bank Lists by Status ── */}
      <Section title={`Defaulted Banks (${defaultBanks.length})`} icon={AlertTriangle} defaultOpen={defaultBanks.length <= 30}>
        {defaultBanks.length === 0 ? (
          <p className="text-xs text-text-muted">No defaulted banks.</p>
        ) : (
          <div className="max-h-64 overflow-auto scrollbar-thin space-y-1">
            {defaultBanks.map((b, i) => (
              <div key={i} className="flex items-center justify-between px-3 py-1.5 rounded-lg bg-crisis-red/5 border border-crisis-red/10">
                <span className="text-xs text-crisis-red font-medium truncate mr-3">{b.name}</span>
                <span className="text-[10px] text-text-muted font-mono whitespace-nowrap">−${(b.loss / 1e9).toFixed(1)}B</span>
              </div>
            ))}
          </div>
        )}
      </Section>

      {distressedBanks.length > 0 && (
        <Section title={`Distressed Banks (${distressedBanks.length})`} icon={Flame} defaultOpen={distressedBanks.length <= 20}>
          <div className="max-h-64 overflow-auto scrollbar-thin space-y-1">
            {distressedBanks.map((b, i) => (
              <div key={i} className="flex items-center justify-between px-3 py-1.5 rounded-lg bg-amber-500/5 border border-amber-500/10">
                <span className="text-xs text-amber-400 font-medium truncate mr-3">{b.name}</span>
                <span className="text-[10px] text-text-muted font-mono whitespace-nowrap">−${(b.loss / 1e9).toFixed(1)}B</span>
              </div>
            ))}
          </div>
        </Section>
      )}

      {safeBanks.length > 0 && (
        <Section title={`Safe Banks (${safeBanks.length})`} icon={Users} defaultOpen={false}>
          <div className="max-h-64 overflow-auto scrollbar-thin space-y-1">
            {safeBanks.map((b, i) => (
              <div key={i} className="flex items-center justify-between px-3 py-1.5 rounded-lg bg-stability-green/5 border border-stability-green/10">
                <span className="text-xs text-stability-green font-medium truncate mr-3">{b.name}</span>
                <span className="text-[10px] text-text-muted font-mono whitespace-nowrap">${(b.final / 1e9).toFixed(1)}B eq</span>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* ── Intraday Timeline Charts ── */}
      {timelineData.length > 1 && (
        <Section title="Intraday Contagion Timeline" icon={TrendingDown}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Asset Price Decay */}
            <div>
              <h4 className="text-xs font-[family-name:var(--font-mono)] text-text-muted uppercase tracking-wider mb-3">
                Asset Price (Exponential Decay)
              </h4>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={timelineData}>
                  <defs>
                    <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ff2a6d" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#ff2a6d" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="step" tick={{ fill: "#555566", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis
                    tick={{ fill: "#555566", fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                    domain={[0, 1.05]}
                    width={40}
                  />
                  <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#888" }} />
                  <Area type="monotone" dataKey="price" stroke="#ff2a6d" fill="url(#priceGrad)" strokeWidth={2} name="Price" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Defaults & Distressed Stacked */}
            <div>
              <h4 className="text-xs font-[family-name:var(--font-mono)] text-text-muted uppercase tracking-wider mb-3">
                Defaults & Distressed per Step
              </h4>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={timelineData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="step" tick={{ fill: "#555566", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#555566", fontSize: 10 }} axisLine={false} tickLine={false} width={40} />
                  <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#888" }} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Bar dataKey="defaults" stackId="a" fill="#ff2a6d" name="Defaults" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="distressed" stackId="a" fill="#ffaa00" name="Distressed" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Liquidity Gridlock */}
            <div>
              <h4 className="text-xs font-[family-name:var(--font-mono)] text-text-muted uppercase tracking-wider mb-3">
                Liquidity Gridlock (Failed Payments)
              </h4>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={timelineData}>
                  <defs>
                    <linearGradient id="gridlockGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ffaa00" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#ffaa00" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="step" tick={{ fill: "#555566", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#555566", fontSize: 10 }} axisLine={false} tickLine={false} width={50} />
                  <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#888" }} />
                  <Area
                    type="monotone"
                    dataKey="gridlock"
                    stroke="#ffaa00"
                    fill="url(#gridlockGrad)"
                    strokeWidth={2}
                    name="Failed Payments"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Margin Call Spirals */}
            <div>
              <h4 className="text-xs font-[family-name:var(--font-mono)] text-text-muted uppercase tracking-wider mb-3">
                Margin Call Spirals ($B)
              </h4>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={timelineData}>
                  <defs>
                    <linearGradient id="marginGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#b24bf3" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#b24bf3" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="step" tick={{ fill: "#555566", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#555566", fontSize: 10 }} axisLine={false} tickLine={false} width={50} />
                  <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#888" }} />
                  <Area
                    type="monotone"
                    dataKey="marginCalls"
                    stroke="#b24bf3"
                    fill="url(#marginGrad)"
                    strokeWidth={2}
                    name="Margin Calls ($B)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Combined equity-loss timeline */}
          {equityLossTl.length > 1 && (
            <div className="mt-4">
              <h4 className="text-xs font-[family-name:var(--font-mono)] text-text-muted uppercase tracking-wider mb-3">
                Cumulative Equity Loss ($B)
              </h4>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={timelineData}>
                  <defs>
                    <linearGradient id="eqLossGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ff2a6d" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#ff2a6d" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="step" tick={{ fill: "#555566", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#555566", fontSize: 10 }} axisLine={false} tickLine={false} width={50} />
                  <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#888" }} />
                  <Area
                    type="monotone"
                    dataKey="equityLoss"
                    stroke="#ff2a6d"
                    fill="url(#eqLossGrad)"
                    strokeWidth={2.5}
                    name="Equity Loss ($B)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </Section>
      )}

      {/* ── Top 15 Casualties Table ── */}
      <Section title="Top 15 Casualties" icon={AlertTriangle}>
        <div className="overflow-auto">
          <table className="w-full text-xs font-[family-name:var(--font-mono)]">
            <thead>
              <tr className="border-b border-border text-text-muted uppercase tracking-wider">
                <th className="text-left py-2 px-3">#</th>
                <th className="text-left py-2 px-3">Bank</th>
                <th className="text-right py-2 px-3">Initial Eq</th>
                <th className="text-right py-2 px-3">Final Eq</th>
                <th className="text-right py-2 px-3">Loss</th>
                <th className="text-center py-2 px-3">Status</th>
              </tr>
            </thead>
            <tbody>
              {casualties.map((c, i) => (
                <tr
                  key={i}
                  className={cn(
                    "border-b border-border/50 transition-colors",
                    c.status === "Default" && "bg-crisis-red/5",
                    c.status === "Distressed" && "bg-amber-warn/5"
                  )}
                >
                  <td className="py-2 px-3 text-text-muted">{i + 1}</td>
                  <td className="py-2 px-3 text-text-primary">{c.name}</td>
                  <td className="py-2 px-3 text-right text-text-secondary">
                    {formatUSD(c.initial)}
                  </td>
                  <td className="py-2 px-3 text-right text-text-secondary">
                    {formatUSD(c.final)}
                  </td>
                  <td className="py-2 px-3 text-right font-bold text-crisis-red">
                    {formatUSD(c.loss)}
                  </td>
                  <td className="py-2 px-3 text-center">
                    <span
                      className={cn(
                        "inline-block px-2 py-0.5 rounded-full text-[10px] font-bold",
                        c.status === "Default" && "bg-crisis-red/20 text-crisis-red",
                        c.status === "Distressed" && "bg-amber-warn/20 text-amber-warn",
                        c.status === "Safe" && "bg-stability-green/20 text-stability-green"
                      )}
                    >
                      {c.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

    </>
  );
}

/* ════════════════════════════════════════════════════════════════
   STRATEGIC (GAME THEORY) DETAIL PANEL
   ════════════════════════════════════════════════════════════════ */
function StrategicDetail({ results }) {
  if (!results?.opaque || !results?.transparent) return null;

  const opaque = results.opaque;
  const transparent = results.transparent;
  const capitalSaved = results.capital_saved ?? 0;

  const tlA = opaque.timeline ?? {};
  const tlB = transparent.timeline ?? {};
  const steps = tlA.steps ?? [];

  // Build combined timeline data
  const timelineData = steps.map((s, i) => ({
    step: s,
    lossOpaque: ((tlA.cumulative_fire_sale_loss?.[i] ?? 0) / 1e9),
    lossTransparent: ((tlB.cumulative_fire_sale_loss?.[i] ?? 0) / 1e9),
    beliefOpaque: tlA.avg_belief?.[i] ?? 0,
    beliefTransparent: tlB.avg_belief?.[i] ?? 0,
    runOpaque: (tlA.run_fraction?.[i] ?? 0) * 100,
    runTransparent: (tlB.run_fraction?.[i] ?? 0) * 100,
    runsOpaque: tlA.n_runs?.[i] ?? 0,
    runsTransparent: tlB.n_runs?.[i] ?? 0,
  }));

  return (
    <>
      {/* ── A/B Comparison Header ── */}
      <div className="grid grid-cols-3 gap-4">
        <GlassPanel className="text-center" glow="red">
          <p className="text-[10px] font-[family-name:var(--font-mono)] text-crisis-red uppercase tracking-wider mb-2">
            Opaque Regime (Fog of War)
          </p>
          <p className="text-3xl font-bold font-[family-name:var(--font-mono)] text-crisis-red">
            {((opaque.run_rate ?? 0) * 100).toFixed(1)}%
          </p>
          <p className="text-[10px] text-text-muted mt-1">Bank Run Rate</p>
          <p className="text-lg font-[family-name:var(--font-mono)] text-crisis-red mt-2">
            {formatUSD(opaque.total_fire_sale_loss ?? 0)}
          </p>
          <p className="text-[10px] text-text-muted">Total Fire-Sale Loss</p>
        </GlassPanel>

        <GlassPanel className="text-center gradient-border">
          <p className="text-[10px] font-[family-name:var(--font-mono)] text-data-blue uppercase tracking-wider mb-2">
            Capital Saved by AI Transparency
          </p>
          <p className="text-4xl font-bold font-[family-name:var(--font-mono)] text-data-blue text-glow-blue">
            {formatUSD(capitalSaved)}
          </p>
          <p className="text-[10px] text-text-muted mt-2">Transparency Dividend</p>
        </GlassPanel>

        <GlassPanel className="text-center" glow="green">
          <p className="text-[10px] font-[family-name:var(--font-mono)] text-stability-green uppercase tracking-wider mb-2">
            Transparent Regime (AI Signal)
          </p>
          <p className="text-3xl font-bold font-[family-name:var(--font-mono)] text-stability-green">
            {((transparent.run_rate ?? 0) * 100).toFixed(1)}%
          </p>
          <p className="text-[10px] text-text-muted mt-1">Bank Run Rate</p>
          <p className="text-lg font-[family-name:var(--font-mono)] text-stability-green mt-2">
            {formatUSD(transparent.total_fire_sale_loss ?? 0)}
          </p>
          <p className="text-[10px] text-text-muted">Total Fire-Sale Loss</p>
        </GlassPanel>
      </div>

      {/* ── Theory Explainer ── */}
      <div className="glass rounded-lg p-4 text-xs text-text-secondary leading-relaxed">
        <span className="text-neon-purple font-bold">Morris & Shin (1998):</span>{" "}
        Banks fail from <span className="text-crisis-red font-semibold">coordination failure</span> (panics),
        not just insolvency. When information is opaque, creditors rely on noisy private signals and may
        rationally run even on solvent banks. The AI transparency signal anchors expectations around the true
        fundamental, preventing self-fulfilling crises.
      </div>

      {/* ── Charts ── */}
      {timelineData.length > 1 && (
        <Section title="A/B Timeline Comparison" icon={BarChart3}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Cumulative Fire-Sale Losses */}
            <div>
              <h4 className="text-xs font-[family-name:var(--font-mono)] text-text-muted uppercase tracking-wider mb-3">
                Cumulative Fire-Sale Losses ($B)
              </h4>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={timelineData}>
                  <defs>
                    <linearGradient id="opaqueGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ff2a6d" stopOpacity={0.2} />
                      <stop offset="100%" stopColor="#ff2a6d" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="transGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#05d5fa" stopOpacity={0.2} />
                      <stop offset="100%" stopColor="#05d5fa" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="step" tick={{ fill: "#555566", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#555566", fontSize: 10 }} axisLine={false} tickLine={false} width={50} />
                  <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#888" }} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Area type="monotone" dataKey="lossOpaque" stroke="#ff2a6d" fill="url(#opaqueGrad)" strokeWidth={2.5} name="Opaque" />
                  <Area type="monotone" dataKey="lossTransparent" stroke="#05d5fa" fill="url(#transGrad)" strokeWidth={2.5} name="Transparent" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Average P(Default) Belief */}
            <div>
              <h4 className="text-xs font-[family-name:var(--font-mono)] text-text-muted uppercase tracking-wider mb-3">
                Average P(Default) Belief
              </h4>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={timelineData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="step" tick={{ fill: "#555566", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis
                    tick={{ fill: "#555566", fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                    domain={[0, 1]}
                    width={40}
                  />
                  <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#888" }} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Line type="monotone" dataKey="beliefOpaque" stroke="#ff2a6d" strokeWidth={2} dot={{ r: 3 }} name="Opaque" />
                  <Line type="monotone" dataKey="beliefTransparent" stroke="#05d5fa" strokeWidth={2} dot={{ r: 3 }} name="Transparent" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Withdrawal Rate */}
            <div>
              <h4 className="text-xs font-[family-name:var(--font-mono)] text-text-muted uppercase tracking-wider mb-3">
                Withdrawal Rate per Step (%)
              </h4>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={timelineData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="step" tick={{ fill: "#555566", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis
                    tick={{ fill: "#555566", fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                    domain={[0, 105]}
                    width={40}
                  />
                  <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#888" }} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Bar dataKey="runOpaque" fill="#ff2a6d" name="Opaque %" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="runTransparent" fill="#05d5fa" name="Transparent %" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Number of Runs */}
            <div>
              <h4 className="text-xs font-[family-name:var(--font-mono)] text-text-muted uppercase tracking-wider mb-3">
                Number of Bank Runs per Step
              </h4>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={timelineData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="step" tick={{ fill: "#555566", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#555566", fontSize: 10 }} axisLine={false} tickLine={false} width={40} />
                  <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#888" }} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Bar dataKey="runsOpaque" fill="#ff2a6d" name="Opaque Runs" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="runsTransparent" fill="#05d5fa" name="Transparent Runs" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Section>
      )}

      {/* ── Step-by-step Table ── */}
      {timelineData.length > 1 && (
        <Section title="Step-by-Step Detail" icon={BarChart3} defaultOpen={false}>
          <div className="overflow-auto">
            <table className="w-full text-xs font-[family-name:var(--font-mono)]">
              <thead>
                <tr className="border-b border-border text-text-muted uppercase tracking-wider">
                  <th className="py-2 px-2 text-center">Step</th>
                  <th className="py-2 px-2 text-right text-crisis-red">Runs (O)</th>
                  <th className="py-2 px-2 text-right text-crisis-red">Run% (O)</th>
                  <th className="py-2 px-2 text-right text-crisis-red">P(def) O</th>
                  <th className="py-2 px-2 text-right text-crisis-red">Loss O ($B)</th>
                  <th className="py-2 px-2 text-right text-stability-green">Runs (T)</th>
                  <th className="py-2 px-2 text-right text-stability-green">Run% (T)</th>
                  <th className="py-2 px-2 text-right text-stability-green">P(def) T</th>
                  <th className="py-2 px-2 text-right text-stability-green">Loss T ($B)</th>
                </tr>
              </thead>
              <tbody>
                {timelineData.map((row, i) => (
                  <tr key={i} className="border-b border-border/50">
                    <td className="py-1.5 px-2 text-center text-text-muted">{row.step}</td>
                    <td className="py-1.5 px-2 text-right">{row.runsOpaque}</td>
                    <td className="py-1.5 px-2 text-right">{row.runOpaque.toFixed(0)}%</td>
                    <td className="py-1.5 px-2 text-right">{row.beliefOpaque.toFixed(4)}</td>
                    <td className="py-1.5 px-2 text-right">{row.lossOpaque.toFixed(2)}</td>
                    <td className="py-1.5 px-2 text-right">{row.runsTransparent}</td>
                    <td className="py-1.5 px-2 text-right">{row.runTransparent.toFixed(0)}%</td>
                    <td className="py-1.5 px-2 text-right">{row.beliefTransparent.toFixed(4)}</td>
                    <td className="py-1.5 px-2 text-right">{row.lossTransparent.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* ── Game Event Log ── */}
      {(tlA.decisions?.length > 0 || tlB.decisions?.length > 0) && (
        <Section title="Game Event Log" icon={Flame} defaultOpen={false}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Opaque log */}
            <div>
              <h4 className="text-xs font-[family-name:var(--font-mono)] text-crisis-red uppercase tracking-wider mb-3">
                Opaque Regime
              </h4>
              <div className="space-y-2 max-h-80 overflow-auto scrollbar-thin">
                {(tlA.decisions ?? []).map((stepDecisions, t) => {
                  const nWithdraw = stepDecisions.filter(d => d === "WITHDRAW").length;
                  const nAgents = stepDecisions.length;
                  const stepLoss = tlA.step_fire_sale_loss?.[t] ?? 0;
                  const cumLoss = tlA.cumulative_fire_sale_loss?.[t] ?? 0;
                  // Find flippers from previous step
                  const prevDecisions = t > 0 ? tlA.decisions[t - 1] : null;
                  const flippedToRun = [];
                  const flippedToStay = [];
                  if (prevDecisions) {
                    stepDecisions.forEach((d, i) => {
                      if (d !== prevDecisions[i]) {
                        const name = opaque.agent_names?.[i] ?? `Bank ${i}`;
                        if (d === "WITHDRAW") flippedToRun.push(name);
                        else flippedToStay.push(name);
                      }
                    });
                  }
                  return (
                    <div key={t} className="glass rounded-lg p-3 text-[11px] font-[family-name:var(--font-mono)]">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-text-muted">Step {t + 1}:</span>
                        <span className="text-crisis-red font-bold">{nWithdraw}/{nAgents} withdrew</span>
                        <span className="text-text-muted">·</span>
                        <span className="text-crisis-red">${(stepLoss / 1e9).toFixed(3)}B loss</span>
                        <span className="text-text-muted">·</span>
                        <span className="text-text-secondary">${(cumLoss / 1e9).toFixed(3)}B cumulative</span>
                      </div>
                      {flippedToRun.length > 0 && (
                        <div className="text-crisis-red/90 text-[10px] mt-2 leading-relaxed bg-crisis-red/5 p-2 rounded border border-crisis-red/10">
                          <span className="font-bold uppercase tracking-wider text-[9px] opacity-70 block mb-1">Flipped to WITHDRAW ({flippedToRun.length})</span>
                          {flippedToRun.slice(0, 8).join(", ")}
                          {flippedToRun.length > 8 && <span className="text-crisis-red/60 italic"> ...and {flippedToRun.length - 8} others</span>}
                        </div>
                      )}
                      {flippedToStay.length > 0 && (
                        <div className="text-stability-green/90 text-[10px] mt-2 leading-relaxed bg-stability-green/5 p-2 rounded border border-stability-green/10">
                          <span className="font-bold uppercase tracking-wider text-[9px] opacity-70 block mb-1">Flipped to ROLL OVER ({flippedToStay.length})</span>
                          {flippedToStay.slice(0, 8).join(", ")}
                          {flippedToStay.length > 8 && <span className="text-stability-green/60 italic"> ...and {flippedToStay.length - 8} others</span>}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Transparent log */}
            <div>
              <h4 className="text-xs font-[family-name:var(--font-mono)] text-stability-green uppercase tracking-wider mb-3">
                Transparent Regime
              </h4>
              <div className="space-y-2 max-h-80 overflow-auto scrollbar-thin">
                {(tlB.decisions ?? []).map((stepDecisions, t) => {
                  const nWithdraw = stepDecisions.filter(d => d === "WITHDRAW").length;
                  const nAgents = stepDecisions.length;
                  const stepLoss = tlB.step_fire_sale_loss?.[t] ?? 0;
                  const cumLoss = tlB.cumulative_fire_sale_loss?.[t] ?? 0;
                  const prevDecisions = t > 0 ? tlB.decisions[t - 1] : null;
                  const flippedToRun = [];
                  const flippedToStay = [];
                  if (prevDecisions) {
                    stepDecisions.forEach((d, i) => {
                      if (d !== prevDecisions[i]) {
                        const name = transparent.agent_names?.[i] ?? `Bank ${i}`;
                        if (d === "WITHDRAW") flippedToRun.push(name);
                        else flippedToStay.push(name);
                      }
                    });
                  }
                  return (
                    <div key={t} className="glass rounded-lg p-3 text-[11px] font-[family-name:var(--font-mono)]">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-text-muted">Step {t + 1}:</span>
                        <span className="text-stability-green font-bold">{nWithdraw}/{nAgents} withdrew</span>
                        <span className="text-text-muted">·</span>
                        <span className="text-stability-green">${(stepLoss / 1e9).toFixed(3)}B loss</span>
                        <span className="text-text-muted">·</span>
                        <span className="text-text-secondary">${(cumLoss / 1e9).toFixed(3)}B cumulative</span>
                      </div>
                      {flippedToRun.length > 0 && (
                        <div className="text-crisis-red/90 text-[10px] mt-2 leading-relaxed bg-crisis-red/5 p-2 rounded border border-crisis-red/10">
                          <span className="font-bold uppercase tracking-wider text-[9px] opacity-70 block mb-1">Flipped to WITHDRAW ({flippedToRun.length})</span>
                          {flippedToRun.slice(0, 8).join(", ")}
                          {flippedToRun.length > 8 && <span className="text-crisis-red/60 italic"> ...and {flippedToRun.length - 8} others</span>}
                        </div>
                      )}
                      {flippedToStay.length > 0 && (
                        <div className="text-stability-green/90 text-[10px] mt-2 leading-relaxed bg-stability-green/5 p-2 rounded border border-stability-green/10">
                          <span className="font-bold uppercase tracking-wider text-[9px] opacity-70 block mb-1">Flipped to ROLL OVER ({flippedToStay.length})</span>
                          {flippedToStay.slice(0, 8).join(", ")}
                          {flippedToStay.length > 8 && <span className="text-stability-green/60 italic"> ...and {flippedToStay.length - 8} others</span>}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </Section>
      )}

      {/* ── Decision Heatmap ── */}
      {(tlA.decisions?.length > 0 || tlB.decisions?.length > 0) && (
        <Section title="Per-Agent Decision Heatmap" icon={Users} defaultOpen={false}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Opaque heatmap */}
            <div>
              <h4 className="text-xs font-[family-name:var(--font-mono)] text-crisis-red uppercase tracking-wider mb-3">
                Opaque Regime
              </h4>
              <div className="overflow-auto max-h-96 scrollbar-thin">
                <table className="text-[9px] font-[family-name:var(--font-mono)] border-collapse">
                  <thead>
                    <tr>
                      <th className="py-1 px-2 text-left text-text-muted sticky left-0 bg-void-panel z-10">Agent</th>
                      {steps.map((s) => (
                        <th key={s} className="py-1 px-1 text-center text-text-muted min-w-[28px]">S{s}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Array.from({ length: tlA.decisions?.[0]?.length ?? 0 }, (_, agentIdx) => (
                      <tr key={agentIdx}>
                        <td className="py-0.5 px-2 text-text-secondary whitespace-nowrap sticky left-0 bg-void-panel z-10">
                          {opaque.agent_names?.[agentIdx] ?? `Bank ${agentIdx}`}
                        </td>
                        {steps.map((_, stepIdx) => {
                          const decision = tlA.decisions?.[stepIdx]?.[agentIdx];
                          const prevDecision = stepIdx > 0 ? tlA.decisions?.[stepIdx - 1]?.[agentIdx] : null;
                          const isFlipped = prevDecision != null && prevDecision !== decision;
                          const isWithdraw = decision === "WITHDRAW";
                          return (
                            <td
                              key={stepIdx}
                              title={`${opaque.agent_names?.[agentIdx] ?? `Bank ${agentIdx}`}: ${decision} (Step ${stepIdx + 1})${isFlipped ? " ★ FLIPPED" : ""}`}
                              className={cn(
                                "py-0.5 px-1 text-center",
                                isWithdraw ? "bg-crisis-red/30 text-crisis-red" : "bg-stability-green/20 text-stability-green",
                                isFlipped && "ring-1 ring-white/40"
                              )}
                            >
                              {isWithdraw ? "W" : "R"}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex items-center gap-4 mt-2 text-[9px] text-text-muted font-[family-name:var(--font-mono)]">
                <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-sm bg-crisis-red/30" /> W = Withdraw</span>
                <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-sm bg-stability-green/20" /> R = Roll Over</span>
                <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-sm ring-1 ring-white/40" /> Flipped</span>
              </div>
            </div>

            {/* Transparent heatmap */}
            <div>
              <h4 className="text-xs font-[family-name:var(--font-mono)] text-stability-green uppercase tracking-wider mb-3">
                Transparent Regime
              </h4>
              <div className="overflow-auto max-h-96 scrollbar-thin">
                <table className="text-[9px] font-[family-name:var(--font-mono)] border-collapse">
                  <thead>
                    <tr>
                      <th className="py-1 px-2 text-left text-text-muted sticky left-0 bg-void-panel z-10">Agent</th>
                      {steps.map((s) => (
                        <th key={s} className="py-1 px-1 text-center text-text-muted min-w-[28px]">S{s}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Array.from({ length: tlB.decisions?.[0]?.length ?? 0 }, (_, agentIdx) => (
                      <tr key={agentIdx}>
                        <td className="py-0.5 px-2 text-text-secondary whitespace-nowrap sticky left-0 bg-void-panel z-10">
                          {transparent.agent_names?.[agentIdx] ?? `Bank ${agentIdx}`}
                        </td>
                        {steps.map((_, stepIdx) => {
                          const decision = tlB.decisions?.[stepIdx]?.[agentIdx];
                          const prevDecision = stepIdx > 0 ? tlB.decisions?.[stepIdx - 1]?.[agentIdx] : null;
                          const isFlipped = prevDecision != null && prevDecision !== decision;
                          const isWithdraw = decision === "WITHDRAW";
                          return (
                            <td
                              key={stepIdx}
                              title={`${transparent.agent_names?.[agentIdx] ?? `Bank ${agentIdx}`}: ${decision} (Step ${stepIdx + 1})${isFlipped ? " ★ FLIPPED" : ""}`}
                              className={cn(
                                "py-0.5 px-1 text-center",
                                isWithdraw ? "bg-crisis-red/30 text-crisis-red" : "bg-stability-green/20 text-stability-green",
                                isFlipped && "ring-1 ring-white/40"
                              )}
                            >
                              {isWithdraw ? "W" : "R"}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex items-center gap-4 mt-2 text-[9px] text-text-muted font-[family-name:var(--font-mono)]">
                <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-sm bg-crisis-red/30" /> W = Withdraw</span>
                <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-sm bg-stability-green/20" /> R = Roll Over</span>
                <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-sm ring-1 ring-white/40" /> Flipped</span>
              </div>
            </div>
          </div>
        </Section>
      )}
    </>
  );
}

/* ════════════════════════════════════════════════════════════════
   MAIN EXPORT — FULLSCREEN MODAL
   ════════════════════════════════════════════════════════════════ */
export default function DetailedAnalysis({ open, onClose, results, tab }) {
  if (!open) return null;

  const [llmLoading, setLlmLoading] = useState(false);
  const [llmError, setLlmError] = useState(null);
  const [llmResponse, setLlmResponse] = useState(null);

  const runType = tab === "strategic" ? "game" : tab;

  const handleExplain = async () => {
    if (!results?.run_id) return;
    setLlmLoading(true);
    setLlmError(null);
    try {
      const res = await explainRun({
        runId: results.run_id,
        runType,
        question: "Summarize the results and explain the key graph trends in simple terms.",
      });
      setLlmResponse(res.response);
    } catch (e) {
      setLlmError(e.message);
    } finally {
      setLlmLoading(false);
    }
  };

  const isGame = tab === "strategic";
  const title = isGame
    ? "Strategic Simulation — Global Games A/B Test"
    : tab === "climate"
      ? "Green Swan — Climate Transition Risk Analysis"
      : "Mechanical Simulation — Detailed Analysis";

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-start justify-center"
        >
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-void/80 backdrop-blur-sm"
            onClick={onClose}
          />

          {/* Modal body */}
          <motion.div
            initial={{ y: 40, opacity: 0, scale: 0.97 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            exit={{ y: 40, opacity: 0, scale: 0.97 }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
            data-lenis-prevent
            className="relative z-10 mt-8 mb-8 w-[95vw] max-w-7xl max-h-[90vh] overflow-auto rounded-2xl glass-bright border border-border-bright shadow-2xl scrollbar-thin"
          >
            {/* Header */}
            <div className="sticky top-0 z-10 flex items-center justify-between px-6 py-4 border-b border-border bg-void-panel/90 backdrop-blur-md rounded-t-2xl">
              <div className="flex items-center gap-3">
                <BarChart3 className="h-5 w-5 text-stability-green" />
                <h2 className="text-lg font-[family-name:var(--font-display)] font-bold text-text-primary">
                  {title}
                </h2>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleExplain}
                  disabled={!results?.run_id || llmLoading}
                  className={cn(
                    "px-3 py-2 rounded-lg text-[11px] font-bold font-[family-name:var(--font-mono)] uppercase tracking-wider border transition-colors",
                    llmLoading
                      ? "bg-white/5 text-text-muted border-border"
                      : "bg-neon-purple/10 text-neon-purple border-neon-purple/30 hover:bg-neon-purple/20"
                  )}
                >
                  {llmLoading ? "Generating…" : "Explain"}
                </button>
                <button
                  onClick={onClose}
                  className="p-2 rounded-lg hover:bg-white/5 transition-colors text-text-muted hover:text-white"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="p-6 space-y-6">
              {llmError && (
                <div className="text-xs text-crisis-red bg-crisis-red/10 border border-crisis-red/20 rounded-lg p-3">
                  {llmError}
                </div>
              )}
              {llmResponse && (
                <GlassPanel className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-text-primary">AI Summary</h3>
                    <span className="text-[10px] text-text-muted font-[family-name:var(--font-mono)]">
                      Confidence: {Math.round((llmResponse.confidence ?? 0) * 100)}%
                    </span>
                  </div>
                  <p className="text-sm text-text-secondary">{llmResponse.summary}</p>
                  {Array.isArray(llmResponse.key_points) && llmResponse.key_points.length > 0 && (
                    <ul className="list-disc list-inside text-xs text-text-secondary space-y-1">
                      {llmResponse.key_points.map((pt, i) => (
                        <li key={i}>{pt}</li>
                      ))}
                    </ul>
                  )}
                  {llmResponse.limitations && (
                    <p className="text-[11px] text-text-muted">Limitations: {llmResponse.limitations}</p>
                  )}
                </GlassPanel>
              )}
              {!results ? (
                <div className="text-center py-20 text-text-muted text-sm">
                  Run a simulation first to see detailed analysis.
                </div>
              ) : isGame ? (
                <StrategicDetail results={results} />
              ) : (
                <MechanicalDetail results={results} />
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
