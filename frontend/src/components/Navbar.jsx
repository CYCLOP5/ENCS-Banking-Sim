import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  Network,
  BookOpen,
  Layers,
  Database,
  Menu,
  X,
  GraduationCap,
} from "lucide-react";
import { cn } from "../lib/utils";

const links = [
  { to: "/", label: "Home", icon: Activity },
  { to: "/methodology", label: "Methodology", icon: BookOpen },
  { to: "/implementation", label: "Architecture", icon: Layers },
  { to: "/simulation", label: "Simulation", icon: Network },
  { to: "/benchmarking", label: "Benchmarking", icon: Activity },
  { to: "/banks", label: "Banks", icon: Database },
  { to: "/how-to-use", label: "How to Use", icon: BookOpen },
  { to: "/terminology", label: "Glossary", icon: GraduationCap },
];

export default function Navbar() {
  const { pathname } = useLocation();
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={cn(
        "fixed top-0 inset-x-0 z-50 transition-all duration-500",
        scrolled
          ? "glass-bright border-b border-border shadow-lg shadow-black/40"
          : "bg-transparent"
      )}
    >
      <nav className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-3 group">
          <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-crisis-red/10 border border-crisis-red/20 group-hover:glow-red transition-shadow">
            <Activity className="h-5 w-5 text-crisis-red" />
          </div>
          <div className="flex flex-col">
            <span className="font-[family-name:var(--font-display)] text-lg font-bold tracking-tight text-white leading-none group-hover:text-crisis-red transition-colors">
              ENCS
            </span>
            <span className="text-text-secondary font-medium text-[10px] uppercase tracking-wider leading-none mt-1">
              Systemic Risk Engine
            </span>
          </div>
        </Link>

        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-1">
          {links.map(({ to, label, icon: Icon }) => {
            const active = pathname === to;
            return (
              <Link
                key={to}
                to={to}
                className={cn(
                  "relative flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200 whitespace-nowrap",
                  active
                    ? "text-white"
                    : "text-text-secondary hover:text-text-primary hover:bg-surface-hover"
                )}
              >
                <Icon className="h-4 w-4" />
                {label}
                {active && (
                  <motion.div
                    layoutId="nav-pill"
                    className="absolute inset-0 rounded-lg bg-white/[0.06] border border-border-bright"
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  />
                )}
              </Link>
            );
          })}
        </div>



        {/* Mobile toggle */}
        <button
          className="md:hidden p-2 text-text-secondary"
          onClick={() => setMobileOpen((o) => !o)}
        >
          {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </nav>

      {/* Mobile menu */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="md:hidden glass-bright border-b border-border overflow-hidden"
          >
            <div className="flex flex-col gap-1 p-4">
              {links.map(({ to, label, icon: Icon }) => (
                <Link
                  key={to}
                  to={to}
                  onClick={() => setMobileOpen(false)}
                  className={cn(
                    "flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors whitespace-nowrap",
                    pathname === to
                      ? "bg-surface text-white"
                      : "text-text-secondary hover:text-white hover:bg-surface-hover"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {label}
                </Link>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
