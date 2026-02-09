import { cn } from "../lib/utils";

/**
 * Glassmorphism panel — the primary container used across all HUD elements.
 */
export default function GlassPanel({
  children,
  className,
  glow,
  noPadding = false,
  ...props
}) {
  return (
    <div
      className={cn(
        "glass rounded-2xl",
        !noPadding && "p-5",
        glow === "red" && "glow-red",
        glow === "green" && "glow-green",
        glow === "blue" && "glow-blue",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}
