import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import {
  Network,
  BrainCircuit,
  Gamepad2,
  CloudLightning,
  ArrowDown,
} from "lucide-react";
import GlassPanel from "../components/GlassPanel";
import katex from "katex";

/* ── KaTeX helper ──────────────────────────────────────────────── */
function Tex({ children, display = false }) {
  const html = katex.renderToString(children, {
    throwOnError: false,
    displayMode: display,
  });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

function TexBlock({ children }) {
  return (
    <div className="my-6 overflow-x-auto">
      <Tex display>{children}</Tex>
    </div>
  );
}

/* ── Fade wrapper ──────────────────────────────────────────────── */
function FadeIn({ children, delay = 0, className = "" }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.65, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

/* ── Section heading ───────────────────────────────────────────── */
function SectionHead({ icon: Icon, label, title, color }) {
  const colorMap = {
    green: "text-stability-green bg-stability-green/10 border-stability-green/20",
    blue: "text-data-blue bg-data-blue/10 border-data-blue/20",
    purple: "text-neon-purple bg-neon-purple/10 border-neon-purple/20",
    red: "text-crisis-red bg-crisis-red/10 border-crisis-red/20",
  };
  return (
    <div className="flex items-center gap-3 mb-6">
      <div
        className={`flex h-10 w-10 items-center justify-center rounded-xl border ${colorMap[color]}`}
      >
        <Icon className="h-5 w-5" />
      </div>
      <div>
        <p className="text-[10px] font-[family-name:var(--font-mono)] uppercase tracking-[0.2em] text-text-muted">
          {label}
        </p>
        <h2 className="font-[family-name:var(--font-display)] text-2xl font-bold">{title}</h2>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   METHODOLOGY PAGE
   ═══════════════════════════════════════════════════════════════════ */

export default function Methodology() {
  return (
    <div className="pt-24 pb-20 px-6">
      <div className="mx-auto max-w-3xl">
        {/* Page header */}
        <FadeIn>
          <p className="text-xs font-[family-name:var(--font-mono)] uppercase tracking-[0.25em] text-stability-green mb-4">
            Technical Documentation
          </p>
          <h1 className="font-[family-name:var(--font-display)] text-4xl sm:text-5xl font-bold mb-4 leading-tight">
            Methodology
          </h1>
          <p className="text-text-secondary text-lg leading-relaxed mb-16">
            A deep-dive into the four computational layers powering the ENCS
            engine: clearing physics, graph intelligence, strategic game theory,
            and climate contagion modelling.
          </p>
        </FadeIn>

        {/* ── A. EISENBERG-NOE ─────────────────────────────────────── */}
        <FadeIn>
          <GlassPanel className="mb-12">
            <SectionHead
              icon={Network}
              label="Layer 1 — Physics"
              title="Eisenberg-Noe Clearing"
              color="green"
            />

            <p className="text-text-secondary leading-relaxed mb-4">
              The interbank network is an{" "}
              <span className="text-white">N × N weighted directed graph</span>,
              where entry <Tex>{"W_{ij}"}</Tex> represents what bank{" "}
              <Tex>{"i"}</Tex> owes bank <Tex>{"j"}</Tex>. The total obligation
              of bank <Tex>{"i"}</Tex> is:
            </p>

            <TexBlock>{"\\bar{p}_i = \\sum_{j=1}^{N} W_{ij}"}</TexBlock>

            <p className="text-text-secondary leading-relaxed mb-4">
              Each bank has <b className="text-white">external assets</b>{" "}
              <Tex>{"e_i"}</Tex> and receives payments from counterparties. The
              clearing vector <Tex>{"\\mathbf{p}^*"}</Tex> satisfies:
            </p>

            <TexBlock>
              {
                "p_i^* = \\min\\!\\left(\\bar{p}_i,\\; e_i + \\sum_{j=1}^{N} \\frac{W_{ji}}{\\bar{p}_j}\\, p_j^*\\right)"
              }
            </TexBlock>

            <p className="text-text-secondary leading-relaxed mb-4">
              This fixed-point is computed via <b className="text-white">Picard iteration</b>{" "}
              (Fictitious Default Algorithm). Banks that cannot meet full obligations
              are classified as:
            </p>

            <div className="grid grid-cols-3 gap-3 mt-4">
              {[
                { status: "Safe", color: "text-stability-green", desc: "p* = p̄" },
                { status: "Distressed", color: "text-amber-warn", desc: "Equity < 5%" },
                { status: "Default", color: "text-crisis-red", desc: "p* < p̄" },
              ].map(({ status, color, desc }) => (
                <div key={status} className="glass rounded-lg p-3 text-center">
                  <span className={`text-sm font-bold font-[family-name:var(--font-mono)] ${color}`}>
                    {status}
                  </span>
                  <p className="text-[11px] text-text-muted mt-1">{desc}</p>
                </div>
              ))}
            </div>

            <p className="text-text-secondary leading-relaxed mt-6 text-sm">
              The intraday extension adds <b className="text-white">fire-sale price impact</b>{" "}
              (<Tex>{"\\alpha"}</Tex>), <b className="text-white">panic withdrawal</b>{" "}
              (<Tex>{"\\rho"}</Tex>), and <b className="text-white">margin spirals</b> that
              amplify the cascade across discrete time-steps, implemented in Rust
              for performance.
            </p>

            <div className="mt-6 glass rounded-lg p-4 border border-stability-green/10">
              <h4 className="text-sm font-bold text-stability-green font-[family-name:var(--font-mono)] uppercase tracking-wider mb-3">
                Liquidity Spirals
              </h4>
              <p className="text-text-secondary text-sm leading-relaxed mb-3">
                The Rust engine explicitly calculates <b className="text-white">variation margin calls</b>{" "}
                when asset prices drop. For each bank <Tex>{"i"}</Tex>:
              </p>
              <TexBlock>
                {"M_i = \\text{Exposure}_i \\cdot (1 - P_t) \\cdot \\delta_{\\text{sens}}"}
              </TexBlock>
              <p className="text-text-secondary text-sm leading-relaxed">
                If external assets are insufficient to meet the margin call, the bank
                is forced to <b className="text-crisis-red">liquidate assets</b>, driving
                prices further down and triggering additional margin calls across the
                network — a self-reinforcing liquidity spiral.
              </p>
            </div>
          </GlassPanel>
        </FadeIn>

        {/* ── B. GNN ───────────────────────────────────────────────── */}
        <FadeIn delay={0.1}>
          <GlassPanel className="mb-12">
            <SectionHead
              icon={BrainCircuit}
              label="Layer 2 — AI"
              title="Graph Neural Network"
              color="blue"
            />

            <p className="text-text-secondary leading-relaxed mb-4">
              A <b className="text-white">3-layer GCN</b> (Graph Convolutional Network)
              learns to predict bank default risk from the interbank topology.
              The architecture narrows sequentially{" "}
              (<Tex>{"64 \\to 32 \\to 16"}</Tex> channels) to force
              high-level feature abstraction at each hop. Each
              node has a <b className="text-white">7-dimensional feature vector</b>:
            </p>

            <div className="glass rounded-lg p-4 mb-4 font-[family-name:var(--font-mono)] text-sm">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 text-text-secondary">
                <div>
                  <span className="text-data-blue">0.</span> log(total_assets)
                </div>
                <div>
                  <span className="text-data-blue">1.</span> leverage_ratio
                </div>
                <div>
                  <span className="text-data-blue">2.</span> log(out_strength + 1)
                </div>
                <div>
                  <span className="text-data-blue">3.</span> log(in_strength + 1)
                </div>
                <div>
                  <span className="text-data-blue">4.</span> interbank_ratio
                </div>
                <div>
                  <span className="text-data-blue">5.</span> log(equity_capital + 1)
                </div>
                <div>
                  <span className="text-data-blue">6.</span> log(deriv_notional + 1)
                </div>
              </div>
            </div>

            <p className="text-text-secondary leading-relaxed mb-4">
              The message-passing rule at layer <Tex>{"l"}</Tex>:
            </p>

            <TexBlock>
              {
                "\\mathbf{h}_i^{(l+1)} = \\sigma\\!\\left(\\sum_{j \\in \\mathcal{N}(i)} \\frac{1}{\\sqrt{d_i d_j}} \\, \\mathbf{W}^{(l)} \\mathbf{h}_j^{(l)}\\right)"
              }
            </TexBlock>

            <p className="text-text-secondary leading-relaxed text-sm">
              Trained on <b className="text-white">500 Monte Carlo simulation runs</b> across
              three regimes (Calm / Moderate / Stressed). Binary cross-entropy loss
              optimises for distinguishing Safe vs. Risky (Default + Distressed)
              nodes.
            </p>

            <div className="mt-6 glass rounded-lg p-4 border border-data-blue/10">
              <h4 className="text-sm font-bold text-data-blue font-[family-name:var(--font-mono)] uppercase tracking-wider mb-3">
                Training Details
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm text-text-secondary">
                <div className="glass rounded p-2 text-center">
                  <span className="text-data-blue font-[family-name:var(--font-mono)]">Loss</span>
                  <p className="text-[11px] mt-1">Class-weighted cross-entropy (auto-balanced, capped at 5.0×)</p>
                </div>
                <div className="glass rounded p-2 text-center">
                  <span className="text-data-blue font-[family-name:var(--font-mono)]">Scheduler</span>
                  <p className="text-[11px] mt-1">Cosine Annealing LR over 80 epochs</p>
                </div>
                <div className="glass rounded p-2 text-center">
                  <span className="text-data-blue font-[family-name:var(--font-mono)]">Regularisation</span>
                  <p className="text-[11px] mt-1">Dropout <Tex>{"p = 0.3"}</Tex> between GCN layers</p>
                </div>
              </div>
            </div>
          </GlassPanel>
        </FadeIn>

        {/* ── C. GAME THEORY ───────────────────────────────────────── */}
        <FadeIn delay={0.15}>
          <GlassPanel className="mb-12">
            <SectionHead
              icon={Gamepad2}
              label="Layer 3 — Strategic"
              title="Morris & Shin Global Games"
              color="purple"
            />

            <p className="text-text-secondary leading-relaxed mb-4">
              Banks decide to <b className="text-white">Roll Over</b> (stay) or{" "}
              <b className="text-white">Withdraw</b> (run) based on expected
              utility:
            </p>

            <TexBlock>{"U_i = \\mathbb{E}[\\text{Return}] - \\lambda_i \\cdot \\text{Risk}"}</TexBlock>

            <div className="glass rounded-lg p-4 mb-4 text-sm">
              <p className="text-text-secondary mb-2">Concretely, the engine evaluates two payoffs per agent per step:</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="glass rounded p-3">
                  <span className="text-stability-green font-[family-name:var(--font-mono)] text-xs">Roll Over</span>
                  <TexBlock>{"U_{\\text{stay}} = (1-p)(1+r) + pR - \\lambda\\sigma"}</TexBlock>
                </div>
                <div className="glass rounded p-3">
                  <span className="text-crisis-red font-[family-name:var(--font-mono)] text-xs">Withdraw</span>
                  <TexBlock>{"U_{\\text{run}} = 1 + m"}</TexBlock>
                </div>
              </div>
              <p className="text-text-muted text-xs mt-2">
                Where <Tex>{"r"}</Tex> = interest rate, <Tex>{"R"}</Tex> = recovery rate,{" "}
                and <Tex>{"m"}</Tex> = <b className="text-white">CCP margin pressure</b> — a term that
                rises with market-wide volatility and accumulated fire-sale damage,
                coupling the clearing layer directly into strategic decisions.
              </p>
            </div>

            <p className="text-text-secondary leading-relaxed mb-4">
              Each agent receives a <b className="text-white">private signal</b>{" "}
              <Tex>{"x_i = \\theta + \\varepsilon_i"}</Tex> and a{" "}
              <b className="text-white">public signal</b> <Tex>{"y"}</Tex>. The
              Bayesian posterior for the true state <Tex>{"\\theta"}</Tex>:
            </p>

            <TexBlock>
              {
                "\\mu_{\\text{post}} = \\frac{\\alpha \\cdot y + \\beta \\cdot x_i}{\\alpha + \\beta}, \\quad \\sigma^2_{\\text{post}} = \\frac{1}{\\alpha + \\beta}"
              }
            </TexBlock>

            <p className="text-text-secondary leading-relaxed mb-4">
              Where <Tex>{"\\alpha"}</Tex> = public precision and <Tex>{"\\beta"}</Tex>{" "}
              = private precision. The probability of default:
            </p>

            <TexBlock>
              {
                "P(\\theta < 0) = \\Phi\\!\\left(\\frac{-\\mu_{\\text{post}}}{\\sigma_{\\text{post}}}\\right)"
              }
            </TexBlock>

            <div className="grid grid-cols-2 gap-3 mt-6">
              <div className="glass rounded-lg p-4">
                <p className="text-xs font-[family-name:var(--font-mono)] text-crisis-red uppercase tracking-wider mb-1">
                  Opaque Regime
                </p>
                <p className="text-text-secondary text-sm">
                  Public signal uninformative → agents rely on noisy private
                  signals → coordination failure → self-fulfilling panics.
                </p>
              </div>
              <div className="glass rounded-lg p-4">
                <p className="text-xs font-[family-name:var(--font-mono)] text-stability-green uppercase tracking-wider mb-1">
                  Transparent Regime
                </p>
                <p className="text-text-secondary text-sm">
                  GNN provides accurate public signal → beliefs converge →
                  coordination succeeds → capital saved.
                </p>
              </div>
            </div>

            <div className="mt-6 glass rounded-lg p-4 border border-neon-purple/10">
              <h4 className="text-sm font-bold text-neon-purple font-[family-name:var(--font-mono)] uppercase tracking-wider mb-3">
                Self-Fulfilling Feedback Loop
              </h4>
              <p className="text-text-secondary text-sm leading-relaxed">
                Aggregate withdrawals trigger <b className="text-white">fire-sale losses</b>{" "}
                that degrade the effective solvency signal seen by all agents at the
                next time-step:
              </p>
              <TexBlock>
                {"\\theta_{\\text{eff}}^{(t)} = \\theta \\cdot \\left(1 - 0.5 \\cdot \\frac{\\text{Cum. Loss}}{\\text{Remaining Exposure}}\\right)"}
              </TexBlock>
              <p className="text-text-secondary text-sm leading-relaxed">
                This creates a <b className="text-crisis-red">self-reinforcing death spiral</b>:
                runs erode asset value → signals worsen → more agents withdraw → further
                losses. The transparent regime breaks this cycle by anchoring beliefs
                before the loop can take hold.
              </p>
            </div>
          </GlassPanel>
        </FadeIn>

        {/* ── D. CLIMATE ───────────────────────────────────────────── */}
        <FadeIn delay={0.2}>
          <GlassPanel className="mb-12">
            <SectionHead
              icon={CloudLightning}
              label="Layer 4 — Climate"
              title='Green Swan Transition Risk'
              color="red"
            />

            <p className="text-text-secondary leading-relaxed mb-4">
              Models a sudden <b className="text-white">carbon tax</b> or regulatory shift.
              Each bank has a <b className="text-white">carbon score</b>{" "}
              <Tex>{"c_i \\in [0,1]"}</Tex> reflecting fossil-fuel exposure:
            </p>

            <div className="glass rounded-lg p-4 mb-4 text-sm">
              <div className="grid grid-cols-2 gap-2">
                <div className="text-text-secondary">
                  <span className="text-stability-green font-[family-name:var(--font-mono)]">US banks:</span>{" "}
                  <Tex>{"c \\sim N(0.60, 0.12)"}</Tex>
                </div>
                <div className="text-text-secondary">
                  <span className="text-data-blue font-[family-name:var(--font-mono)]">EU banks:</span>{" "}
                  <Tex>{"c \\sim N(0.30, 0.10)"}</Tex>
                </div>
              </div>
            </div>

            <p className="text-text-secondary leading-relaxed mb-4">
              The net capital shock for bank <Tex>{"i"}</Tex>:
            </p>

            <TexBlock>
              {
                "\\Delta e_i = -\\underbrace{0.20 \\cdot A_i \\cdot c_i \\cdot \\tau}_{\\text{Brown loss}} + \\underbrace{0.15 \\cdot A_i \\cdot (1-c_i) \\cdot s}_{\\text{Green gain}}"
              }
            </TexBlock>

            <p className="text-text-secondary leading-relaxed text-sm">
              Where <Tex>{"\\tau"}</Tex> = carbon tax severity and{" "}
              <Tex>{"s"}</Tex> = green subsidy. The net shock propagates through
              the interbank network via the Eisenberg-Noe engine, demonstrating
              that climate risk is <b className="text-crisis-red">systemic</b>:
              even "green" EU banks can default when their brown US counterparties
              fail.
            </p>

            <div className="mt-4 glass rounded-lg p-3 text-xs text-text-muted border border-crisis-red/10">
              <b className="text-crisis-red">Greenwashing discount:</b> Top 30
              banks reduce reported carbon scores by 10% — but it's not enough
              to prevent cascade when the shock hits.
            </div>
          </GlassPanel>
        </FadeIn>

        {/* References */}
        <FadeIn delay={0.25}>
          <div className="text-xs text-text-muted space-y-1 font-[family-name:var(--font-mono)]">
            <p className="text-text-secondary font-semibold mb-2">References</p>
            <p>
              Eisenberg, L. & Noe, T. (2001). "Systemic Risk in Financial
              Systems." <i>Management Science</i>, 47(2).
            </p>
            <p>
              Morris, S. & Shin, H.S. (1998). "Unique Equilibrium in a Model of
              Self-Fulfilling Currency Attacks." <i>AER</i>, 88(3).
            </p>
            <p>
              Bolton, P. et al. (2020). "The Green Swan." BIS / Banque de France.
            </p>
          </div>
        </FadeIn>
      </div>
    </div>
  );
}
