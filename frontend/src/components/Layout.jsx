import { useEffect, useRef } from "react";
import { Outlet, useLocation } from "react-router-dom";
import Lenis from "lenis";
import Navbar from "./Navbar";

export default function Layout() {
  const lenisRef = useRef(null);
  const { pathname } = useLocation();

  useEffect(() => {
    const lenis = new Lenis({
      lerp: 0.07,
      duration: 1.2,
      smoothWheel: true,
      wheelMultiplier: 0.8,
    });
    lenisRef.current = lenis;

    function raf(time) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }
    requestAnimationFrame(raf);

    return () => {
      lenis.destroy();
    };
  }, []);

  // Scroll to top on route change
  useEffect(() => {
    lenisRef.current?.scrollTo(0, { immediate: true });
  }, [pathname]);

  return (
    <div className="relative min-h-screen bg-void text-text-primary">
      {/* Ambient grid */}
      <div className="pointer-events-none fixed inset-0 grid-bg opacity-40" />

      {/* Nav */}
      <Navbar />

      {/* Page content */}
      <main>
        <Outlet />
      </main>
    </div>
  );
}
