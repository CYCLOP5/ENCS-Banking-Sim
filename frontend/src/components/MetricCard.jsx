import { cn, formatUSD } from "../lib/utils";

/**
 * A single metric display card with optional delta.
 */
export default function MetricCard({
  label,
  value,
  delta,
  prefix,
  suffix,
  color = "text-stability-green",
  className,
  reversed = false, // If true, positive delta is BAD (Red), negative is GOOD (Green). Default false means Green up.
}) {
  const nValue = Number(value);
  const isCurrency =
    typeof value === "number" || (!Number.isNaN(nValue) && value !== "");

  const useUSD = isCurrency && label !== "Defaults" && label !== "Distressed" && !suffix && !prefix;

  const displayVal = useUSD
      ? formatUSD(value)
      : `${prefix ?? ""}${value}${suffix ?? ""}`;

  return (
    <div
      className={cn(
        "glass rounded-xl p-4 flex flex-col gap-1 min-w-[140px]",
        className
      )}
    >
      <span className="text-[11px] font-medium uppercase tracking-widest text-text-muted font-[family-name:var(--font-mono)]">
        {label}
      </span>
      <div className="flex items-baseline gap-2">
        <span
          className={cn(
            "text-2xl font-bold tracking-tight font-[family-name:var(--font-mono)]",
            color
          )}
        >
          {displayVal}
        </span>
        {delta !== undefined && delta !== 0 && (
          <span
            className={cn(
              "text-xs font-[family-name:var(--font-mono)]",
              (delta > 0 && reversed) || (delta < 0 && !reversed) 
                ? "text-crisis-red" 
                : "text-stability-green"
            )}
          >
            {delta > 0 ? "↑" : "↓"} {useUSD ? formatUSD(Math.abs(delta)).replace("$", "") : Math.abs(delta).toLocaleString()}
          </span>
        )}
      </div>
    </div>
  );
}
