import { motion } from "framer-motion";
import { cn } from "../lib/utils";

/**
 * Glowing feature card for the Landing page.
 */
export default function FeatureCard({ icon: Icon, title, description, color, delay = 0 }) {
  const colorMap = {
    red: {
      border: "border-crisis-red/20 hover:border-crisis-red/40",
      glow: "hover:glow-red",
      icon: "text-crisis-red bg-crisis-red/10",
      accent: "text-crisis-red",
    },
    green: {
      border: "border-stability-green/20 hover:border-stability-green/40",
      glow: "hover:glow-green",
      icon: "text-stability-green bg-stability-green/10",
      accent: "text-stability-green",
    },
    blue: {
      border: "border-data-blue/20 hover:border-data-blue/40",
      glow: "hover:glow-blue",
      icon: "text-data-blue bg-data-blue/10",
      accent: "text-data-blue",
    },
    purple: {
      border: "border-neon-purple/20 hover:border-neon-purple/40",
      glow: "hover:shadow-[0_0_20px_rgba(178,75,243,0.15)]",
      icon: "text-neon-purple bg-neon-purple/10",
      accent: "text-neon-purple",
    },
  };

  const c = colorMap[color] || colorMap.green;

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.6, delay }}
      whileHover={{ y: -4 }}
      className={cn(
        "glass rounded-2xl p-6 border transition-all duration-300 cursor-default",
        c.border,
        c.glow
      )}
    >
      <div
        className={cn(
          "flex h-11 w-11 items-center justify-center rounded-xl mb-4",
          c.icon
        )}
      >
        <Icon className="h-5 w-5" />
      </div>
      <h3
        className={cn(
          "font-[family-name:var(--font-display)] text-lg font-semibold mb-2",
          c.accent
        )}
      >
        {title}
      </h3>
      <p className="text-sm text-text-secondary leading-relaxed">{description}</p>
    </motion.div>
  );
}
