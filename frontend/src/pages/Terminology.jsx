import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  X,
  Landmark,
  TrendingDown,
  HeartPulse,
  Shield,
  BrainCircuit,
  Users,
  CloudLightning,
  Scale,
  Scissors,
  ArrowDownUp,
  Banknote,
  BarChart3,
  AlertTriangle,
  Network,
} from "lucide-react";
import GlassPanel from "../components/GlassPanel";
import katex from "katex";

// ── LaTeX helper ─────────────────────────────────────────────────
function Tex({ children, display = false }) {
  const html = katex.renderToString(children, {
    throwOnError: false,
    displayMode: display,
  });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

// ── Color palette per category ──────────────────────────────────
const CATEGORY_STYLE = {
  Physics:     { accent: "text-crisis-red",      border: "border-crisis-red/20",      bg: "bg-crisis-red/5",      glow: "red"   },
  "Market Risk": { accent: "text-crisis-red",    border: "border-crisis-red/20",      bg: "bg-crisis-red/5",      glow: "red"   },
  Status:      { accent: "text-stability-green",  border: "border-stability-green/20", bg: "bg-stability-green/5", glow: "green" },
  Topology:    { accent: "text-stability-green",  border: "border-stability-green/20", bg: "bg-stability-green/5", glow: "green" },
  AI:          { accent: "text-data-blue",         border: "border-data-blue/20",       bg: "bg-data-blue/5",       glow: "blue"  },
  "Game Theory": { accent: "text-neon-purple",    border: "border-neon-purple/20",     bg: "bg-neon-purple/5",     glow: null    },
  Climate:     { accent: "text-stability-green",  border: "border-stability-green/20", bg: "bg-stability-green/5", glow: "green" },
  Risk:        { accent: "text-crisis-red",       border: "border-crisis-red/20",      bg: "bg-crisis-red/5",      glow: "red"   },
  Regulation:  { accent: "text-data-blue",        border: "border-data-blue/20",       bg: "bg-data-blue/5",       glow: "blue"  },
};

// ── Term Definitions ────────────────────────────────────────────
const TERMS = [
  {
    id: "encs",
    title: "Eisenberg-Noe Clearing",
    category: "Physics",
    icon: Landmark,
    simple:
      'The "laws of physics" for paying back debt. If Bank A can\'t pay Bank B, Bank B might not be able to pay Bank C. It\'s a chain reaction of IOUs.',
    technical:
      "A fixed-point algorithm that computes the clearing payment vector p* such that no bank pays more than it has (limited liability) and equity is strictly non-negative. The system converges via Picard iteration on the lattice of payment vectors.",
    math: "p^* = \\min\\bigl(\\bar{p},\\; e + \\Pi^\\top p^*\\bigr)",
  },
  {
    id: "firesale",
    title: "Fire-Sale Spiral",
    category: "Market Risk",
    icon: TrendingDown,
    simple:
      "When banks sell assets in a panic, prices crash. This forces other banks to sell, crashing prices further — a death spiral.",
    technical:
      "Asset price decays exponentially based on sold volume. This creates a non-linear positive feedback loop where mark-to-market losses trigger further liquidations.",
    math: "P_{t+1} = P_t \\cdot e^{-\\alpha \\cdot V_t / 10^{12}}",
  },
  {
    id: "status",
    title: "Distressed vs. Default",
    category: "Status",
    icon: HeartPulse,
    simple:
      '"Default" means you have $0 left. "Distressed" means you still have money, but you lost so much that you\'re a zombie bank — alive but barely functioning.',
    technical:
      "Default: Equity < 0 (insolvent, cannot meet obligations). Distressed: Equity ratio drops below the distress_threshold (e.g., losing >50% of initial capital), triggering early warning signals and potential bank runs.",
    math: "\\text{Default: } E_i < 0 \\qquad \\text{Distressed: } \\frac{E_i}{E_i^0} < \\theta_{\\text{distress}}",
  },
  {
    id: "ccp",
    title: "Central Counterparty (CCP)",
    category: "Topology",
    icon: Shield,
    simple:
      'A "Super-Bank" that sits in the middle of everyone. If Bank A fails, the CCP absorbs the hit so Bank B stays safe. It\'s a financial firewall.',
    technical:
      "Transforms the N×N bilateral topology into a Hub-and-Spoke graph. The CCP holds a Default Fund funded by member margins to absorb shockwaves. Reduces systemic interconnectedness at the cost of concentrated risk.",
    math: "W_{\\text{CCP}} = \\sum_{i} m_i \\cdot r_{\\text{default\\_fund}}",
  },
  {
    id: "gnn",
    title: "Graph Neural Network (GNN)",
    category: "AI",
    icon: BrainCircuit,
    simple:
      "An AI that looks at the spiderweb of bank connections to predict who will fail before it happens — like a financial early warning radar.",
    technical:
      "A 3-layer Graph Convolutional Network (GCN) that aggregates neighbor features. It learns non-linear patterns in leverage and interbank exposure to predict binary risk probabilities for each node.",
    math: "H^{(l+1)} = \\sigma\\!\\bigl(\\tilde{D}^{-\\frac{1}{2}} \\tilde{A} \\tilde{D}^{-\\frac{1}{2}} H^{(l)} W^{(l)}\\bigr)",
  },
  {
    id: "morris-shin",
    title: "Morris-Shin Bank Run",
    category: "Game Theory",
    icon: Users,
    simple:
      "A self-fulfilling prophecy. Even if a bank is healthy, if everyone thinks others will pull money out, they all pull out, and the bank dies.",
    technical:
      "A global game where agents receive noisy signals about solvency. If the signal is below a threshold x*, the dominant strategy is to Withdraw, causing a coordination failure and liquidity collapse.",
    math: "x^* = \\theta^* + \\sigma \\Phi^{-1}\\!\\left(\\frac{\\theta^* - c}{\\theta^*}\\right)",
  },
  {
    id: "green-swan",
    title: "Green Swan",
    category: "Climate",
    icon: CloudLightning,
    simple:
      'A sudden climate disaster (or Carbon Tax) that makes "safe" fossil-fuel assets worthless overnight — the climate version of a Black Swan.',
    technical:
      'A "Climate Value-at-Risk" (CVaR) shock applied to the external_assets vector. It specifically penalizes banks with high carbon-intensity scores, creating stranded assets.',
    math: "E_i' = E_i - \\tau_{\\text{carbon}} \\cdot c_i \\cdot A_i^{\\text{ext}}",
  },
  {
    id: "leverage",
    title: "Leverage Ratio",
    category: "Risk",
    icon: Scale,
    simple:
      "How much a bank is borrowing compared to what it actually owns. High leverage = high risk, like building a skyscraper on a thin foundation.",
    technical:
      "The ratio of total assets to equity capital. A leverage ratio of 30× means the bank has $1 of equity for every $30 of assets. A 3.3% drop in asset values wipes out all equity.",
    math: "\\text{Leverage} = \\frac{\\text{Total Assets}}{\\text{Equity}} = \\frac{A_i}{E_i}",
  },
  {
    id: "haircut",
    title: "Fire-Sale Haircut",
    category: "Market Risk",
    icon: Scissors,
    simple:
      "The discount you take when selling assets in an emergency. Instead of getting $100, you might only get $80 — that missing $20 is the haircut.",
    technical:
      "The percentage loss on asset liquidation during distressed sales. Applied to the mark-to-market value of collateral, haircuts increase during market stress due to reduced liquidity and wider bid-ask spreads.",
    math: "V_{\\text{realized}} = V_{\\text{market}} \\cdot (1 - h) \\quad \\text{where } h \\in [0, 1]",
  },
  {
    id: "bilateral-netting",
    title: "Bilateral Netting",
    category: "Topology",
    icon: ArrowDownUp,
    simple:
      'If Bank A owes Bank B $100, and Bank B owes Bank A $70, instead of two payments they just settle the $30 difference. It\'s financial "canceling out."',
    technical:
      "Reduces gross exposures to net positions between counterparty pairs, lowering settlement risk and required liquidity. The bilateral liability matrix is compressed: net(A,B) = max(0, L_AB - L_BA).",
    math: "L_{ij}^{\\text{net}} = \\max(0,\\; L_{ij} - L_{ji})",
  },
  {
    id: "margin-calls",
    title: "Margin Calls",
    category: "Market Risk",
    icon: Banknote,
    simple:
      "When your losses mount, your lender demands you put up more cash immediately. If you can't, they seize your assets — triggering a cascade.",
    technical:
      "Variation margin demands triggered when portfolio losses breach maintenance thresholds. In the simulation, margin calls create an intraday liquidity drain that amplifies fire-sale spirals.",
    math: "M_t = \\sum_{j} \\max\\!\\bigl(0,\\; \\Delta V_{ij,t} - \\text{threshold}_j\\bigr)",
  },
  {
    id: "systemic-risk",
    title: "Systemic Risk",
    category: "Risk",
    icon: AlertTriangle,
    simple:
      "The risk that one bank's failure brings down the entire financial system — like one domino toppling thousands. It's why regulators lose sleep.",
    technical:
      "The risk of cascading failures through interconnected financial institutions. Measured by the fraction of total system equity destroyed and the number of defaults triggered by a single institution's failure.",
    math: "\\text{SR}_i = \\frac{\\sum_{j \\neq i} \\Delta E_j \\mid \\text{fail}(i)}{\\sum_{j} E_j^0}",
  },
  {
    id: "contagion",
    title: "Contagion Channel",
    category: "Physics",
    icon: Network,
    simple:
      "The path that financial distress travels through the system — like a virus spreading from bank to bank through their interbank loans.",
    technical:
      "Contagion propagates via three channels: (1) direct credit loss from counterparty default, (2) fire-sale externalities from asset price depression, and (3) liquidity hoarding from precautionary behavior.",
    math: "\\Delta E_j = -\\sum_{i \\in \\mathcal{D}} L_{ji} \\cdot (1 - R_i) - \\alpha \\cdot \\Delta P \\cdot A_j^{\\text{ext}}",
  },
  {
    id: "debtrank",
    title: "DebtRank",
    category: "AI",
    icon: BarChart3,
    simple:
      "A score that measures how much economic value is destroyed when a specific bank fails. High DebtRank = that bank is a ticking time bomb for the whole system.",
    technical:
      "An iterative algorithm inspired by PageRank that quantifies systemic impact. Each node's distress propagates proportionally to bilateral exposures, creating a cascade metric bounded in [0, 1].",
    math: "h_j^{(t+1)} = \\min\\!\\left(1,\\; h_j^{(t)} + \\sum_{i} W_{ji} \\cdot h_i^{(t)}\\right)",
  },
];

// ── Component ────────────────────────────────────────────────────

export default function Terminology() {
  const [search, setSearch] = useState("");
  const [expandedId, setExpandedId] = useState(null);

  const filtered = TERMS.filter((t) => {
    const q = search.toLowerCase();
    return (
      t.title.toLowerCase().includes(q) ||
      t.category.toLowerCase().includes(q) ||
      t.simple.toLowerCase().includes(q)
    );
  });

  const categories = [...new Set(TERMS.map((t) => t.category))];

  return (
    <div className="min-h-screen pt-24 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-12"
      >
        <h1 className="text-4xl sm:text-5xl font-bold font-display text-text-primary mb-3">
          Terminology
        </h1>
        <p className="text-text-secondary max-w-2xl mx-auto text-lg">
          An interactive glossary of the financial engineering, AI, and game theory
          concepts powering AEGIS.
        </p>
      </motion.div>

      {/* Search */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="max-w-xl mx-auto mb-10"
      >
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-4 w-4 text-text-muted" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search terms…  e.g. fire-sale, GNN, contagion"
            className="w-full rounded-xl border border-white/10 bg-white/[0.03] pl-11 pr-10 py-3 text-sm text-text-primary placeholder:text-text-muted focus:border-stability-green/40 focus:outline-none focus:ring-1 focus:ring-stability-green/20 backdrop-blur-sm transition-colors"
          />
          {search && (
            <button
              onClick={() => setSearch("")}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded-md hover:bg-white/10 text-text-muted"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
        {search && (
          <p className="text-xs text-text-muted mt-2 ml-1">
            {filtered.length} term{filtered.length !== 1 ? "s" : ""} found
          </p>
        )}
      </motion.div>

      {/* Category pills */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.15 }}
        className="flex flex-wrap gap-2 justify-center mb-10"
      >
        <button
          onClick={() => setSearch("")}
          className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
            !search
              ? "border-stability-green/40 bg-stability-green/10 text-stability-green"
              : "border-white/10 bg-white/[0.03] text-text-muted hover:border-white/20"
          }`}
        >
          All
        </button>
        {categories.map((cat) => {
          const style = CATEGORY_STYLE[cat] || CATEGORY_STYLE.Risk;
          const active = search.toLowerCase() === cat.toLowerCase();
          return (
            <button
              key={cat}
              onClick={() => setSearch(active ? "" : cat)}
              className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
                active
                  ? `${style.border} ${style.bg} ${style.accent}`
                  : "border-white/10 bg-white/[0.03] text-text-muted hover:border-white/20"
              }`}
            >
              {cat}
            </button>
          );
        })}
      </motion.div>

      {/* Bento Grid */}
      <div className="columns-1 sm:columns-2 lg:columns-3 gap-4 space-y-4">
        <AnimatePresence mode="popLayout">
          {filtered.map((term, i) => {
            const style = CATEGORY_STYLE[term.category] || CATEGORY_STYLE.Risk;
            const Icon = term.icon;
            const isExpanded = expandedId === term.id;

            return (
              <motion.div
                key={term.id}
                layout
                layoutId={term.id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ type: "spring", damping: 25, stiffness: 300, delay: i * 0.03 }}
                className="break-inside-avoid"
              >
                <GlassPanel
                  className={`cursor-pointer transition-all duration-300 hover:border-white/15 ${
                    isExpanded ? `${style.border} border` : ""
                  }`}
                  glow={isExpanded ? style.glow : undefined}
                  onClick={() => setExpandedId(isExpanded ? null : term.id)}
                >
                  {/* Header */}
                  <div className="flex items-start gap-3 mb-3">
                    <div className={`p-2 rounded-lg ${style.bg} shrink-0`}>
                      <Icon className={`h-5 w-5 ${style.accent}`} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <h3 className="font-semibold text-text-primary text-sm leading-tight">
                          {term.title}
                        </h3>
                        <span
                          className={`text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded ${style.bg} ${style.accent}`}
                        >
                          {term.category}
                        </span>
                      </div>
                    </div>
                    <motion.div
                      animate={{ rotate: isExpanded ? 45 : 0 }}
                      className="text-text-muted shrink-0 mt-0.5"
                    >
                      <X className="h-3.5 w-3.5" />
                    </motion.div>
                  </div>

                  {/* Simple definition — always visible */}
                  <p className="text-text-secondary text-sm leading-relaxed">
                    {term.simple}
                  </p>

                  {/* Expanded content */}
                  <AnimatePresence>
                    {isExpanded && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ type: "spring", damping: 25, stiffness: 300 }}
                        className="overflow-hidden"
                      >
                        <div className="mt-4 pt-4 border-t border-white/5">
                          {/* Technical */}
                          <h4 className={`text-xs font-mono uppercase tracking-wider ${style.accent} mb-2`}>
                            Technical Definition
                          </h4>
                          <p className="text-text-secondary text-sm leading-relaxed mb-4">
                            {term.technical}
                          </p>

                          {/* Math */}
                          {term.math && (
                            <div className="rounded-lg bg-black/30 border border-white/5 px-4 py-3 overflow-x-auto">
                              <div className="text-center">
                                <Tex display>{term.math}</Tex>
                              </div>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </GlassPanel>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      {/* Empty state */}
      {filtered.length === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-20"
        >
          <Search className="h-10 w-10 text-text-muted mx-auto mb-4 opacity-40" />
          <p className="text-text-muted text-sm">
            No terms match "<span className="text-text-primary">{search}</span>"
          </p>
        </motion.div>
      )}
    </div>
  );
}
