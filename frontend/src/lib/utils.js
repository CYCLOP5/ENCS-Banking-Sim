import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

/**
 * Format a dollar value into a human-readable string.
 */
export function formatUSD(value) {
  const abs = Math.abs(value);
  if (abs >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  return `$${value.toLocaleString()}`;
}

/**
 * Risk-score → colour class mapper.
 */
export function riskColor(score) {
  if (score >= 0.7) return "text-crisis-red";
  if (score >= 0.4) return "text-amber-warn";
  return "text-stability-green";
}

export function riskBg(score) {
  if (score >= 0.7) return "bg-crisis-red/15 text-crisis-red border-crisis-red/30";
  if (score >= 0.4) return "bg-amber-warn/15 text-amber-warn border-amber-warn/30";
  return "bg-stability-green/15 text-stability-green border-stability-green/30";
}
