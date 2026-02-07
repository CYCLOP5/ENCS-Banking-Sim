import { useState, useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import {
  Search,
  ArrowUpDown,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Building2,
} from "lucide-react";
import GlassPanel from "../components/GlassPanel";
import { fetchBanks } from "../services/api";
import { cn, formatUSD, riskBg } from "../lib/utils";

const PAGE_SIZE = 25;

function Badge({ children, className }) {
  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-bold font-[family-name:var(--font-mono)] border",
        className
      )}
    >
      {children}
    </span>
  );
}

export default function BankExplorer() {
  const [banks, setBanks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState("total_assets");
  const [sortDir, setSortDir] = useState("desc");
  const [page, setPage] = useState(0);

  useEffect(() => {
    fetchBanks()
      .then((data) => setBanks(data.banks || []))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // Filter + sort
  const filtered = useMemo(() => {
    let data = banks;
    if (search) {
      const q = search.toLowerCase();
      data = data.filter(
        (b) =>
          b.name.toLowerCase().includes(q) ||
          b.region.toLowerCase().includes(q) ||
          b.bank_id.toLowerCase().includes(q)
      );
    }
    data = [...data].sort((a, b) => {
      const av = a[sortKey] ?? 0;
      const bv = b[sortKey] ?? 0;
      return sortDir === "asc" ? av - bv : bv - av;
    });
    return data;
  }, [banks, search, sortKey, sortDir]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const pageData = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const toggleSort = (key) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
    setPage(0);
  };

  const columns = [
    { key: "name", label: "Bank", width: "flex-1 min-w-[200px]" },
    { key: "region", label: "Region", width: "w-20" },
    { key: "tier", label: "Tier", width: "w-24" },
    { key: "total_assets", label: "Total Assets", width: "w-32" },
    { key: "equity", label: "Equity", width: "w-28" },
    { key: "leverage_ratio", label: "Leverage", width: "w-24" },
    { key: "gnn_risk_score", label: "GNN Risk", width: "w-28" },
    { key: "carbon_score", label: "Carbon", width: "w-24" },
  ];

  return (
    <div className="pt-20 pb-10 px-4 sm:px-6 min-h-screen">
      <div className="mx-auto max-w-[1400px]">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6"
        >
          <div className="flex items-center gap-3 mb-2">
            <Building2 className="h-5 w-5 text-stability-green" />
            <h1 className="font-[family-name:var(--font-display)] text-3xl font-bold">
              Bank Explorer
            </h1>
          </div>
          <p className="text-text-secondary text-sm">
            Browse {banks.length} institutions across the global interbank
            network. Fuzzy search by name, sort by any metric.
          </p>
        </motion.div>

        {/* Search + stats bar */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="flex flex-col sm:flex-row items-start sm:items-center gap-3 mb-4"
        >
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-text-muted" />
            <input
              type="text"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(0);
              }}
              placeholder="Search by name, region, ID..."
              className="w-full h-10 pl-10 pr-4 rounded-xl glass text-sm text-white placeholder:text-text-muted focus:outline-none focus:ring-1 focus:ring-stability-green/30"
            />
          </div>
          <div className="flex items-center gap-4 text-xs text-text-muted font-[family-name:var(--font-mono)]">
            <span>
              {filtered.length} results · Page {page + 1}/{totalPages || 1}
            </span>
          </div>
        </motion.div>

        {/* Table */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <GlassPanel noPadding className="overflow-hidden">
            {loading ? (
              <div className="flex items-center justify-center py-20">
                <Loader2 className="h-6 w-6 animate-spin text-stability-green" />
              </div>
            ) : error ? (
              <div className="p-8 text-center text-crisis-red text-sm">
                {error}
              </div>
            ) : (
              <>
                {/* Header */}
                <div className="flex items-center gap-0 border-b border-border px-4 py-2.5 bg-white/[0.02]">
                  {columns.map(({ key, label, width }) => (
                    <button
                      key={key}
                      onClick={() => toggleSort(key)}
                      className={cn(
                        "flex items-center gap-1 text-[10px] uppercase tracking-wider font-bold transition-colors",
                        width,
                        sortKey === key
                          ? "text-stability-green"
                          : "text-text-muted hover:text-text-secondary"
                      )}
                    >
                      {label}
                      {sortKey === key && (
                        <ArrowUpDown className="h-3 w-3" />
                      )}
                    </button>
                  ))}
                </div>

                {/* Rows */}
                {pageData.map((bank, i) => (
                  <div
                    key={bank.id}
                    className={cn(
                      "flex items-center gap-0 px-4 py-3 border-b border-border/50 transition-colors hover:bg-surface-hover",
                      i % 2 === 0 && "bg-white/[0.01]"
                    )}
                  >
                    {/* Name */}
                    <div className="flex-1 min-w-[200px] pr-2">
                      <span className="text-sm font-medium text-white truncate block">
                        {bank.name}
                      </span>
                      <span className="text-[10px] text-text-muted font-[family-name:var(--font-mono)]">
                        {bank.bank_id}
                      </span>
                    </div>
                    {/* Region */}
                    <div className="w-20">
                      <Badge
                        className={
                          bank.region === "US"
                            ? "bg-data-blue/10 text-data-blue border-data-blue/20"
                            : bank.region === "EU"
                            ? "bg-stability-green/10 text-stability-green border-stability-green/20"
                            : "bg-amber-warn/10 text-amber-warn border-amber-warn/20"
                        }
                      >
                        {bank.region}
                      </Badge>
                    </div>
                    {/* Tier */}
                    <div className="w-24">
                      <span
                        className={cn(
                          "text-xs font-[family-name:var(--font-mono)]",
                          bank.tier === "super_core"
                            ? "text-crisis-red"
                            : bank.tier === "core"
                            ? "text-amber-warn"
                            : "text-text-muted"
                        )}
                      >
                        {bank.tier?.replace("_", " ")}
                      </span>
                    </div>
                    {/* Total Assets */}
                    <div className="w-32 text-sm font-[family-name:var(--font-mono)] text-white">
                      {formatUSD(bank.total_assets)}
                    </div>
                    {/* Equity */}
                    <div className="w-28 text-sm font-[family-name:var(--font-mono)] text-text-secondary">
                      {formatUSD(bank.equity)}
                    </div>
                    {/* Leverage */}
                    <div className="w-24 text-sm font-[family-name:var(--font-mono)] text-text-secondary">
                      {(bank.leverage_ratio || 0).toFixed(1)}%
                    </div>
                    {/* GNN Risk */}
                    <div className="w-28">
                      <Badge className={riskBg(bank.gnn_risk_score)}>
                        {(bank.gnn_risk_score * 100).toFixed(1)}%
                      </Badge>
                    </div>
                    {/* Carbon */}
                    <div className="w-24">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-stability-green to-crisis-red"
                            style={{
                              width: `${(bank.carbon_score || 0) * 100}%`,
                            }}
                          />
                        </div>
                        <span className="text-[10px] font-[family-name:var(--font-mono)] text-text-muted w-8 text-right">
                          {((bank.carbon_score || 0) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}

                {/* Pagination */}
                <div className="flex items-center justify-between px-4 py-3">
                  <button
                    onClick={() => setPage((p) => Math.max(0, p - 1))}
                    disabled={page === 0}
                    className="flex items-center gap-1 text-xs text-text-muted hover:text-white disabled:opacity-30 transition-colors"
                  >
                    <ChevronLeft className="h-4 w-4" />
                    Previous
                  </button>
                  <div className="flex gap-1">
                    {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                      const p =
                        totalPages <= 7
                          ? i
                          : page < 4
                          ? i
                          : page > totalPages - 4
                          ? totalPages - 7 + i
                          : page - 3 + i;
                      return (
                        <button
                          key={p}
                          onClick={() => setPage(p)}
                          className={cn(
                            "h-7 w-7 rounded-md text-xs font-[family-name:var(--font-mono)] transition-colors",
                            p === page
                              ? "bg-stability-green/20 text-stability-green border border-stability-green/30"
                              : "text-text-muted hover:text-white hover:bg-surface"
                          )}
                        >
                          {p + 1}
                        </button>
                      );
                    })}
                  </div>
                  <button
                    onClick={() =>
                      setPage((p) => Math.min(totalPages - 1, p + 1))
                    }
                    disabled={page >= totalPages - 1}
                    className="flex items-center gap-1 text-xs text-text-muted hover:text-white disabled:opacity-30 transition-colors"
                  >
                    Next
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              </>
            )}
          </GlassPanel>
        </motion.div>
      </div>
    </div>
  );
}
