import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, TrendingDown, AlertTriangle, ChevronDown, Search } from "lucide-react";
import GlassPanel from "../components/GlassPanel";
import MetricCard from "../components/MetricCard";
import { fetchBanks, runSimulation } from "../services/api";
import { cn, formatUSD } from "../lib/utils";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

/* ── Searchable Select Component ──────────────────────────────── */
function SearchableSelect({ value, onChange, options, placeholder = "Select..." }) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const ref = useRef(null);

  useEffect(() => {
    function handleClickOutside(event) {
      if (ref.current && !ref.current.contains(event.target)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const filteredOptions = options.filter((o) =>
    o.label.toLowerCase().includes(search.toLowerCase())
  );

  const selectedLabel = options.find((o) => o.value === value)?.label || placeholder;

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => {
          setOpen(!open);
          setSearch(""); // Reset search on open
        }}
        className="w-full flex items-center justify-between pl-4 pr-3 py-3 rounded-lg bg-surface-dark border border-white/10 text-sm text-white hover:bg-white/[0.04] transition-colors focus:outline-none focus:border-secondary/50 group"
      >
        <span className="truncate mr-2 font-[family-name:var(--font-mono)]">{selectedLabel}</span>
        <ChevronDown className={cn("w-4 h-4 text-text-muted transition-transform group-hover:text-white", open ? "rotate-180" : "rotate-0")} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -5 }}
            className="absolute top-full left-0 right-0 mt-2 max-h-80 overflow-hidden rounded-lg bg-[#0B0B15] border border-white/10 shadow-2xl z-50 flex flex-col"
          >
            <div className="p-2 border-b border-white/5 sticky top-0 bg-[#0B0B15] z-10">
              <div className="relative">
                <Search className="absolute left-2.5 top-2.5 w-3.5 h-3.5 text-text-muted" />
                <input
                  autoFocus
                  type="text"
                  placeholder="Search banks..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="w-full bg-white/5 border border-transparent rounded-md pl-8 pr-3 py-2 text-xs text-white placeholder-text-muted focus:bg-white/10 focus:outline-none focus:ring-1 focus:ring-secondary/50 font-[family-name:var(--font-mono)]"
                />
              </div>
            </div>

            <div className="overflow-y-auto flex-1 p-1 custom-scrollbar">
              {filteredOptions.length === 0 ? (
                <div className="px-3 py-4 text-center text-xs text-text-muted italic font-[family-name:var(--font-mono)]">
                  No banks found.
                </div>
              ) : (
                filteredOptions.map((option) => (
                  <button
                    key={option.value}
                    onClick={() => {
                      onChange(option.value);
                      setOpen(false);
                    }}
                    className={cn(
                      "w-full text-left px-3 py-2 text-xs font-[family-name:var(--font-mono)] transition-colors rounded-md truncate",
                      option.value === value
                        ? "bg-secondary/10 text-secondary"
                        : "text-text-secondary hover:bg-white/[0.04] hover:text-text-primary"
                    )}
                  >
                    {option.label}
                  </button>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function Benchmarking() {
  const [banks, setBanks] = useState([]);
  const [selectedBankIdx, setSelectedBankIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState({ smart: null, uniform: null });

  useEffect(() => {
    fetchBanks().then((data) => setBanks(data.banks)).catch(console.error);
  }, []);

  const processResult = (res) => {
    if (!res) return null;

    // Extract defaults from status array
    const defaultedBanks = [];
    if (res.status && res.bank_names) {
      res.status.forEach((status, idx) => {
        if (status === "Default" && res.bank_names[idx]) {
          defaultedBanks.push(res.bank_names[idx]);
        }
      });
    }

    return {
      defaults: {
        count: res.n_defaults ?? 0,
        banks: defaultedBanks
      },
      losses: {
        total_capital_lost: res.equity_loss ?? 0
      }
    };
  };

  const runComparison = async () => {
    setLoading(true);
    try {
      // Run both simulations in parallel
      // Note: we use camelCase as expected by api.js, which maps to snake_case for backend
      const [smartRes, uniformRes] = await Promise.all([
        runSimulation({
          triggerIdx: selectedBankIdx,
          topologyType: "smart",
          severity: 1.0,
        }),
        runSimulation({
          triggerIdx: selectedBankIdx,
          topologyType: "uniform",
          severity: 1.0,
        }),
      ]);

      setResults({
        smart: processResult(smartRes),
        uniform: processResult(uniformRes)
      });
    } catch (err) {
      console.error("Benchmark failed:", err);
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart data if results exist
  const chartData = results.smart && results.uniform ? [
    {
      name: "Defaults",
      Smart: results.smart.defaults.count,
      Uniform: results.uniform.defaults.count,
    },
    {
      name: "Loss ($B)",
      Smart: results.smart.losses.total_capital_lost / 1e9,
      Uniform: results.uniform.losses.total_capital_lost / 1e9,
    },
  ] : [];

  return (
    <div className="min-h-screen pt-24 pb-12 px-6">
      <div className="max-w-7xl mx-auto space-y-8">

        {/* Header */}
        <div className="space-y-4">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl font-light tracking-tight text-white mb-2"
          >
            Model <span className="text-secondary font-medium">Benchmarking</span>
          </motion.h1>
          <p className="text-text-secondary max-w-2xl text-lg font-light leading-relaxed">
            Directly compare our <span className="text-white">Minimum-Density "Smart" Model</span> against the industry-standard <span className="text-white">Maximum-Entropy "Uniform" Model</span>. See how standard methods fail to capture concentrated risk.
          </p>
        </div>

        {/* Controls */}
        <GlassPanel className="p-6 flex items-end gap-6">
          <div className="space-y-2 flex-grow max-w-md">
            <label className="text-xs text-text-secondary uppercase tracking-wider font-mono">Trigger Bank</label>
            <SearchableSelect
              value={selectedBankIdx}
              onChange={setSelectedBankIdx}
              options={banks.map((b, i) => ({
                value: i,
                label: `${b.name} (${b.ticker}) — $${formatUSD(b.total_assets)}`
              }))}
              placeholder="Select a trigger bank..."
            />
          </div>
          <button
            onClick={runComparison}
            disabled={loading}
            className={cn(
              "flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all shadow-lg hover:shadow-cyan-500/20",
              loading
                ? "bg-white/5 text-white/50 cursor-not-allowed"
                : "bg-stability-green text-void-void hover:bg-stability-green/90"
            )}
          >
            {loading ? (
              <span className="flex items-center gap-2">Processing...</span>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Benchmark
              </>
            )}
          </button>
        </GlassPanel>

        {/* Results Area */}
        {results.smart && results.uniform && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

            {/* Smart Model Column */}
            <div className="space-y-6">
              <div className="flex items-center gap-3 mb-2">
                <div className="grid place-items-center w-8 h-8 rounded-full bg-secondary/20 text-secondary">
                  <span className="text-xs font-bold font-mono text-center w-full block">A</span>
                </div>
                <h2 className="text-xl text-white font-medium">Smart (Core-Periphery)</h2>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <MetricCard
                  title="Defaults"
                  value={results.smart.defaults.count}
                  subValue={`${((results.smart.defaults.count / banks.length) * 100).toFixed(1)}%`}
                  trend="neutral"
                  icon={AlertTriangle}
                />
                <MetricCard
                  title="Capital Lost"
                  value={formatUSD(results.smart.losses.total_capital_lost)}
                  subValue="Total System Equity"
                  trend="down"
                  icon={TrendingDown}
                />
              </div>

              <GlassPanel className="p-4 h-64 border-l-4 border-l-secondary">
                <h3 className="text-sm text-secondary font-mono mb-4 uppercase tracking-wider">Top Failures</h3>
                <div className="space-y-2 overflow-y-auto h-48 pr-2 custom-scrollbar">
                  {results.smart.defaults.banks.map((bank, i) => (
                    <div key={i} className="flex justify-between items-center text-sm p-2 bg-white/5 rounded">
                      <span className="text-white">{bank}</span>
                      <span className="text-crisis-red text-xs">DEFAULT</span>
                    </div>
                  ))}
                  {results.smart.defaults.count === 0 && (
                    <div className="text-text-muted text-sm italic">No defaults triggered.</div>
                  )}
                </div>
              </GlassPanel>
            </div>

            {/* Uniform Model Column */}
            <div className="space-y-6">
              <div className="flex items-center gap-3 mb-2">
                <div className="grid place-items-center w-8 h-8 rounded-full bg-white/10 text-text-muted">
                  <span className="text-xs font-bold font-mono text-center w-full block">B</span>
                </div>
                <h2 className="text-xl text-text-muted font-medium">Standard (Max-Entropy)</h2>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <MetricCard
                  title="Defaults"
                  value={results.uniform.defaults.count}
                  subValue={`${((results.uniform.defaults.count / banks.length) * 100).toFixed(1)}%`}
                  trend="neutral"
                  icon={AlertTriangle}
                  className="opacity-75"
                />
                <MetricCard
                  title="Capital Lost"
                  value={formatUSD(results.uniform.losses.total_capital_lost)}
                  subValue="Total System Equity"
                  trend="down"
                  icon={TrendingDown}
                  className="opacity-75"
                />
              </div>

              <GlassPanel className="p-4 h-64 opacity-75">
                <h3 className="text-sm text-text-muted font-mono mb-4 uppercase tracking-wider">Top Failures</h3>
                <div className="space-y-2 overflow-y-auto h-48 pr-2 custom-scrollbar">
                  {results.uniform.defaults.banks.map((bank, i) => (
                    <div key={i} className="flex justify-between items-center text-sm p-2 bg-white/5 rounded">
                      <span className="text-white">{bank}</span>
                      <span className="text-crisis-red text-xs">DEFAULT</span>
                    </div>
                  ))}
                  {results.uniform.defaults.count === 0 && (
                    <div className="text-text-muted text-sm italic">No defaults triggered.</div>
                  )}
                </div>
              </GlassPanel>
            </div>

            {/* Comparison Chart */}
            <GlassPanel className="col-span-1 lg:col-span-2 p-6 h-[400px] mt-8">
              <h3 className="text-lg text-white font-medium mb-6">Impact Comparison</h3>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" horizontal={false} />
                  <XAxis type="number" stroke="#ffffff50" fontSize={12} tickFormatter={(val) => val.toLocaleString()} />
                  <YAxis dataKey="name" type="category" stroke="#ffffff50" fontSize={12} width={100} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#000000cc', border: '1px solid #ffffff20', borderRadius: '8px' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  <Bar dataKey="Smart" fill="#00f2ff" radius={[0, 4, 4, 0]} barSize={30} />
                  <Bar dataKey="Uniform" fill="#ffffff40" radius={[0, 4, 4, 0]} barSize={30} />
                </BarChart>
              </ResponsiveContainer>
            </GlassPanel>

          </div>
        )}

        {/* Explanation Section */}
        <div className="mt-12 border-t border-white/10 pt-8">
          <h3 className="text-xl text-white font-light mb-6">Why is the difference so large?</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="space-y-2">
              <h4 className="text-secondary font-medium flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-secondary" />
                Smart Model (Model A)
              </h4>
              <p className="text-text-secondary leading-relaxed">
                The <span className="text-white">"Core-Periphery" topology</span> correctly identifies that major banks (SIFIs) are highly interconnected. When a key node fails, the shock propagates rapidly through these dense connections, triggering a cascade (e.g., 13 defaults). This accurately reflects real-world systemic risk.
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="text-text-muted font-medium flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-white/20" />
                Standard Model (Model B)
              </h4>
              <p className="text-text-secondary leading-relaxed">
                The <span className="text-white">"Max-Entropy" topology</span> assumes connections are spread out uniformly and thinly. This artificial dilution "absorbs" the shock, resulting in few defaults (often only the trigger). This demonstrates why standard models fail: they underestimate risk by ignoring network structure.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
