import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { Link } from "react-router-dom";
import {
  Network,
  BrainCircuit,
  Gamepad2,
  CloudLightning,
  ArrowRight,
  ChevronDown,
  Shield,
  Zap,
  TrendingDown,
} from "lucide-react";
import HeroGlobe from "../components/HeroGlobe";
import FeatureCard from "../components/FeatureCard";

/* ── Fade-up animation wrapper ──────────────────────────────────── */
function FadeUp({ children, delay = 0, className = "" }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

export default function Landing() {
  const heroRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"],
  });

  const globeY = useTransform(scrollYProgress, [0, 1], [0, 150]);
  const globeOpacity = useTransform(scrollYProgress, [0, 0.7], [1, 0]);
  const textY = useTransform(scrollYProgress, [0, 1], [0, -60]);

  return (
    <>
      {/* ═══ HERO ═══════════════════════════════════════════════════ */}
      <section
        ref={heroRef}
        className="relative flex min-h-screen items-center justify-center overflow-hidden"
      >
        {/* 3D Globe background */}
        <motion.div
          style={{ y: globeY, opacity: globeOpacity }}
          className="absolute inset-0 flex items-center justify-center pointer-events-none"
        >
          <HeroGlobe className="h-[700px] w-[700px] opacity-70" />
        </motion.div>

        {/* Radial overlay */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_0%,#050505_70%)]" />

        {/* Text content */}
        <motion.div
          style={{ y: textY }}
          className="relative z-10 mx-auto max-w-5xl px-6 text-center"
        >
          {/* Tag */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="inline-flex items-center gap-2 rounded-full border border-crisis-red/20 bg-crisis-red/5 px-4 py-1.5 mb-8"
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute h-full w-full rounded-full bg-crisis-red opacity-75" />
              <span className="relative rounded-full h-2 w-2 bg-crisis-red" />
            </span>
            <span className="text-xs font-[family-name:var(--font-mono)] text-crisis-red tracking-wide uppercase">
              Systemic Risk Intelligence
            </span>
          </motion.div>

          {/* Main heading */}
          <motion.h1
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
            className="font-[family-name:var(--font-display)] text-5xl sm:text-7xl lg:text-8xl font-bold leading-[0.95] tracking-tight"
          >
            <span className="block">The Architecture</span>
            <span className="block mt-2">
              of{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-crisis-red via-neon-purple to-stability-green">
                Collapse
              </span>
            </span>
          </motion.h1>

          {/* Sub */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.6 }}
            className="mt-8 mx-auto max-w-2xl text-lg text-text-secondary leading-relaxed"
          >
            Map the invisible fault-lines in global banking.
            Simulate cascading failures across{" "}
            <span className="text-stability-green font-medium">500+ institutions</span>,
            predict contagion with{" "}
            <span className="text-data-blue font-medium">Graph Neural Networks</span>,
            and stress-test against{" "}
            <span className="text-crisis-red font-medium">Green Swan</span> climate shocks.
          </motion.p>

          {/* CTAs */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
            className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link
              to="/simulation"
              className="group flex items-center gap-2 rounded-xl bg-crisis-red px-7 py-3.5 text-sm font-semibold text-white shadow-lg shadow-crisis-red/20 hover:shadow-crisis-red/30 transition-all hover:scale-105"
            >
              Launch Simulation
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </Link>
            <Link
              to="/methodology"
              className="flex items-center gap-2 rounded-xl border border-border-bright px-7 py-3.5 text-sm font-medium text-text-secondary hover:text-white hover:border-white/20 transition-all"
            >
              Read Methodology
            </Link>
          </motion.div>
        </motion.div>

        {/* Scroll hint */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        >
          <span className="text-[10px] uppercase tracking-[0.2em] text-text-muted font-[family-name:var(--font-mono)]">
            Scroll to explore
          </span>
          <motion.div
            animate={{ y: [0, 6, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            <ChevronDown className="h-4 w-4 text-text-muted" />
          </motion.div>
        </motion.div>
      </section>

      {/* ═══ STATS BAR ═════════════════════════════════════════════ */}
      <section className="relative border-y border-border bg-void-light/50">
        <div className="mx-auto grid max-w-6xl grid-cols-2 md:grid-cols-4 divide-x divide-border">
          {[
            { value: "500+", label: "Banks Modeled", icon: Shield },
            { value: "24T", label: "Interbank Volume", icon: TrendingDown },
            { value: "7-dim", label: "GNN Feature Space", icon: BrainCircuit },
            { value: "<1s", label: "Rust Simulation", icon: Zap },
          ].map(({ value, label, icon: Icon }, i) => (
            <FadeUp key={label} delay={i * 0.1}>
              <div className="flex flex-col items-center gap-2 p-8 md:p-10">
                <Icon className="h-5 w-5 text-text-muted mb-1" />
                <span className="text-3xl font-bold font-[family-name:var(--font-mono)] tracking-tight text-white">
                  {value}
                </span>
                <span className="text-xs text-text-muted uppercase tracking-widest">
                  {label}
                </span>
              </div>
            </FadeUp>
          ))}
        </div>
      </section>

      {/* ═══ FEATURES ══════════════════════════════════════════════ */}
      <section className="py-28 px-6">
        <div className="mx-auto max-w-6xl">
          <FadeUp>
            <p className="text-xs font-[family-name:var(--font-mono)] uppercase tracking-[0.25em] text-stability-green mb-4">
              Engine Modules
            </p>
          </FadeUp>
          <FadeUp delay={0.1}>
            <h2 className="font-[family-name:var(--font-display)] text-4xl sm:text-5xl font-bold mb-16 leading-tight">
              Four Pillars of
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-stability-green to-data-blue">
                Systemic Intelligence
              </span>
            </h2>
          </FadeUp>

          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
            <FeatureCard
              icon={Network}
              title="Eisenberg-Noe Clearing"
              description="Physics-based cascade simulation. Computes unique clearing payment vectors via fixed-point iteration over the interbank liability matrix."
              color="green"
              delay={0}
            />
            <FeatureCard
              icon={BrainCircuit}
              title="Graph Neural Network"
              description="7-dimensional node features trained on 500 Monte Carlo scenarios. Predicts bank risk scores from topology + balance-sheet signals."
              color="blue"
              delay={0.1}
            />
            <FeatureCard
              icon={Gamepad2}
              title="Game Theory Engine"
              description="Morris & Shin (1998) global games. Models coordination failure via Bayesian belief updates under opaque vs. transparent regimes."
              color="purple"
              delay={0.2}
            />
            <FeatureCard
              icon={CloudLightning}
              title="Green Swan Shock"
              description="Climate transition risk module. Simulates sudden carbon tax stranding brown assets, with cross-border US→EU contagion propagation."
              color="red"
              delay={0.3}
            />
          </div>
        </div>
      </section>

      {/* ═══ CTA ═══════════════════════════════════════════════════ */}
      <section className="py-24 px-6">
        <FadeUp>
          <div className="mx-auto max-w-3xl text-center">
            <h2 className="font-[family-name:var(--font-display)] text-3xl sm:text-4xl font-bold mb-6">
              Ready to stress-test the system?
            </h2>
            <p className="text-text-secondary mb-10 max-w-xl mx-auto">
              Enter the simulation dashboard, trigger cascading failures, and
              watch the network unravel in real-time 3D.
            </p>
            <Link
              to="/simulation"
              className="group inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-crisis-red to-neon-purple px-8 py-4 text-sm font-semibold text-white shadow-lg shadow-crisis-red/20 hover:shadow-crisis-red/30 transition-all hover:scale-105"
            >
              Enter Mission Control
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </Link>
          </div>
        </FadeUp>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-10 px-6">
        <div className="mx-auto max-w-6xl flex flex-col sm:flex-row items-center justify-between gap-4">
          <span className="text-xs text-text-muted font-[family-name:var(--font-mono)]">
            ENCS v1.0 — Eisenberg-Noe Contagion Simulation
          </span>
          <span className="text-xs text-text-muted">
            Built for DataHack 2026 &middot; Team LowerTaperFade
          </span>
        </div>
      </footer>
    </>
  );
}
