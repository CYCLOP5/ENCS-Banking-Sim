import { motion } from "framer-motion";
import GlassPanel from "../components/GlassPanel";
import {
    Activity,
    Zap,
    CloudLightning,
    Network,
    Shield,
    Search,
    Users,
    Globe,
    TrendingDown
} from "lucide-react";

export default function HowToUse() {
    return (
        <div className="min-h-screen w-full bg-void-void text-text-primary px-6 py-24 font-[family-name:var(--font-sans)] selection:bg-stability-green/30 selection:text-stability-green">

            {/* Background Gradients */}
            <div className="fixed inset-0 z-0 pointer-events-none">
                <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-data-blue/10 rounded-full blur-[100px] animate-pulse" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-stability-green/10 rounded-full blur-[100px] animate-pulse" style={{ animationDelay: "2s" }} />
            </div>

            <div className="relative z-10 max-w-4xl mx-auto space-y-12">

                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-center space-y-4"
                >
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 mb-4">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-data-blue opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-data-blue"></span>
                        </span>
                        <span className="text-xs font-medium text-data-blue tracking-wider font-[family-name:var(--font-mono)]">USER GUIDE</span>
                    </div>
                    <h1 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white via-white to-white/60">
                        How to Use the Simulator
                    </h1>
                    <p className="text-text-secondary text-lg max-w-2xl mx-auto">
                        A comprehensive guide to the three simulation modes: Mechanical, Strategic, and Climate.
                    </p>
                </motion.div>

                {/* 1. Mechanical Mode */}
                <motion.section
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.1 }}
                >
                    <GlassPanel className="p-8 border-l-4 border-l-crisis-red">
                        <div className="flex items-center gap-4 mb-6">
                            <div className="p-3 rounded-xl bg-crisis-red/10 border border-crisis-red/20 shadow-[0_0_15px_-3px_rgba(255,42,109,0.3)]">
                                <Network className="w-6 h-6 text-crisis-red" />
                            </div>
                            <div>
                                <h2 className="text-2xl font-bold text-white">Mechanical Mode</h2>
                                <p className="text-sm text-crisis-red font-[family-name:var(--font-mono)] uppercase tracking-wider">The "Lehman Moment" Engine</p>
                            </div>
                        </div>

                        <p className="text-text-secondary leading-relaxed mb-6">
                            This mode simulates <strong>direct contagion</strong> through financial networks. It models how a shock to one bank (or asset class) propagates mechanically through interbank lending liabilities.
                            Uses the <strong>Eisenberg-Noe (2001)</strong> clearing vector algorithm to determine simultaneous clearing payments.
                        </p>

                        <div className="grid md:grid-cols-2 gap-4">
                            <div className="p-4 rounded-lg bg-white/5 border border-white/10 user-select-text">
                                <h4 className="text-sm font-bold text-white mb-2 flex items-center gap-2">
                                    <Activity className="w-3 h-3 text-text-muted" /> Key Parameters
                                </h4>
                                <ul className="space-y-2 text-xs text-text-secondary">
                                    <li><strong className="text-white">Severity:</strong> Magnitude of the initial shock (paper loss).</li>
                                    <li><strong className="text-white">Fire Sale α:</strong> Price impact of asset liquidations. Higher α = steeper price drops.</li>
                                    <li><strong className="text-white">Panic Rate:</strong> Rate at which unsecured funding is pulled.</li>
                                </ul>
                            </div>
                            <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                                <h4 className="text-sm font-bold text-white mb-2 flex items-center gap-2">
                                    <Search className="w-3 h-3 text-text-muted" /> What to Look For
                                </h4>
                                <p className="text-xs text-text-secondary">
                                    Watch for <strong>cascading defaults</strong> (red nodes). A small shock can amplify if the network is highly interconnected ("Too Big to Fail") or if Fire Sales degrade the value of common assets.
                                </p>
                            </div>
                        </div>
                    </GlassPanel>
                </motion.section>

                {/* 2. Strategic Mode */}
                <motion.section
                    initial={{ opacity: 0, x: 20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.2 }}
                >
                    <GlassPanel className="p-8 border-l-4 border-l-neon-purple">
                        <div className="flex items-center gap-4 mb-6">
                            <div className="p-3 rounded-xl bg-neon-purple/10 border border-neon-purple/20 shadow-[0_0_15px_-3px_rgba(139,92,246,0.3)]">
                                <Users className="w-6 h-6 text-neon-purple" />
                            </div>
                            <div>
                                <h2 className="text-2xl font-bold text-white">Strategic Mode</h2>
                                <p className="text-sm text-neon-purple font-[family-name:var(--font-mono)] uppercase tracking-wider">Game Theory & Bank Runs</p>
                            </div>
                        </div>

                        <p className="text-text-secondary leading-relaxed mb-6">
                            Models the <strong>psychology</strong> of investors. Based on the <strong>Morris & Shin (1998)</strong> Global Games framework.
                            Investors receive noisy signals about bank health and must decide to <strong>Roll Over</strong> (keep money) or <strong>Withdraw</strong> (run).
                            Compares an "Opaque" world (traditional/fog of war) vs a "Transparent" world (AI-driven clarity).
                        </p>

                        <div className="grid md:grid-cols-2 gap-4">
                            <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                                <h4 className="text-sm font-bold text-white mb-2 flex items-center gap-2">
                                    <Activity className="w-3 h-3 text-text-muted" /> Key Parameters
                                </h4>
                                <ul className="space-y-2 text-xs text-text-secondary">
                                    <li><strong className="text-white">Risk Aversion (λ):</strong> How paranoid investors are. High λ = panic easily.</li>
                                    <li><strong className="text-white">Noise (σ):</strong> Uncertainty in private signals. High noise breaks coordination.</li>
                                    <li><strong className="text-white">Solvency (θ):</strong> True fundamental health of the bank.</li>
                                </ul>
                            </div>
                            <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                                <h4 className="text-sm font-bold text-white mb-2 flex items-center gap-2">
                                    <Search className="w-3 h-3 text-text-muted" /> What to Look For
                                </h4>
                                <p className="text-xs text-text-secondary">
                                    The <strong>Transparency Dividend</strong>. In the "Positive Dividend" scenario, notice how the Opaque regime collapses (red) due to uncertainty, while the Transparent regime (green) survives because the AI proves the banks are solvent.
                                </p>
                            </div>
                        </div>
                    </GlassPanel>
                </motion.section>

                {/* 3. Climate Mode */}
                <motion.section
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.3 }}
                >
                    <GlassPanel className="p-8 border-l-4 border-l-stability-green">
                        <div className="flex items-center gap-4 mb-6">
                            <div className="p-3 rounded-xl bg-stability-green/10 border border-stability-green/20 shadow-[0_0_15px_-3px_rgba(52,211,153,0.3)]">
                                <Globe className="w-6 h-6 text-stability-green" />
                            </div>
                            <div>
                                <h2 className="text-2xl font-bold text-white">Climate Mode</h2>
                                <p className="text-sm text-stability-green font-[family-name:var(--font-mono)] uppercase tracking-wider">Green Swan Events</p>
                            </div>
                        </div>

                        <p className="text-text-secondary leading-relaxed mb-6">
                            Simulates <strong>Transition Risk</strong>. Policies (Carbon Taxes) or technology shifts render "Brown" assets (fossil fuels) worthless, while boosting "Green" assets.
                            Based on <strong>NGFS (Network for Greening the Financial System)</strong> scenarios.
                        </p>

                        <div className="grid md:grid-cols-2 gap-4">
                            <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                                <h4 className="text-sm font-bold text-white mb-2 flex items-center gap-2">
                                    <Activity className="w-3 h-3 text-text-muted" /> Key Parameters
                                </h4>
                                <ul className="space-y-2 text-xs text-text-secondary">
                                    <li><strong className="text-white">Carbon Tax:</strong> Direct penalty on Brown RWA (Risk Weighted Assets).</li>
                                    <li><strong className="text-white">Green Subsidy:</strong> Boost to Green RWA value.</li>
                                    <li><strong className="text-white">Brown/Green Bias:</strong> How much of a bank's portfolio is dirty vs clean.</li>
                                </ul>
                            </div>
                            <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                                <h4 className="text-sm font-bold text-white mb-2 flex items-center gap-2">
                                    <Search className="w-3 h-3 text-text-muted" /> What to Look For
                                </h4>
                                <p className="text-xs text-text-secondary">
                                    <strong>Stranded Assets</strong>. Banks with high "Brown Bias" will suffer massive losses as Carbon Taxes rise. This can trigger a systemic crisis even without a traditional liquidity shock.
                                </p>
                            </div>
                        </div>

                        <div className="mt-6 p-4 rounded-lg bg-orange-500/10 border border-orange-500/20">
                            <h4 className="text-sm font-bold text-orange-400 mb-2 flex items-center gap-2">
                                <TrendingDown className="w-4 h-4" /> Why does "Paris Agreement" cause more defaults than "Hot House"?
                            </h4>
                            <p className="text-xs text-text-secondary leading-relaxed">
                                This simulation models <strong>Transition Risk</strong> (policy impact), not Physical Risk (weather damage).
                                <br /><br />
                                • <strong>Paris Agreement:</strong> High Carbon Tax = Immediate asset write-downs = <span className="text-crisis-red">Financial Shock</span>.
                                <br />
                                • <strong>Hot House World:</strong> No Carbon Tax = Assets keep value (for now) = <span className="text-stability-green">Short-term Stability</span>.
                                <br /><br />
                                The model correctly shows that <em>saving the planet</em> is expensive for banks holding dirty assets, while <em>doing nothing</em> is cheap in the short run.
                            </p>
                        </div>
                    </GlassPanel>
                </motion.section>


            </div>
        </div>
    );
}
