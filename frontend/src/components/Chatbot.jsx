import { useEffect, useMemo, useRef, useState } from "react";
import { MessageSquare, Send, X, Sparkles, Trash2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Rnd } from "react-rnd";
import { motion, AnimatePresence } from "framer-motion";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist";

const Plot = createPlotlyComponent(Plotly);
import GlassPanel from "./GlassPanel";
import { chatWithAssistant } from "../services/api";
import { cn } from "../lib/utils";

const STORAGE_KEY = "encs_chat_messages";
const MAX_MESSAGES = 100;
const POS_KEY = "encs_chat_pos";
const SIZE_KEY = "encs_chat_size";

function tryParseJson(text) {
  if (!text || typeof text !== "string") return null;
  const trimmed = text.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) return null;
  try {
    return JSON.parse(trimmed);
  } catch {
    return null;
  }
}

function normalizeText(text) {
  if (!text || typeof text !== "string") return "";
  const withBreaks = text.replace(/<\s*br\s*\/?>/gi, "\n");
  return withBreaks.replace(/<[^>]+>/g, "");
}

function buildChartFromEvidence(chart, evidence) {
  if (!chart || !evidence) return null;
  if (chart.source === "latest_run.graphs.series" && chart.series_key) {
    const series = evidence?.latest_run?.graphs?.series?.[chart.series_key] || [];
    const kind = ["line", "bar", "scatter", "area", "histogram"].includes(chart.kind)
      ? chart.kind
      : "line";
    const isBar = kind === "bar";
    const isScatter = kind === "scatter";
    const isArea = kind === "area";
    const isHistogram = kind === "histogram";
    return {
      data: [
        {
          x: isHistogram ? undefined : series.map((_, i) => i + 1),
          y: isHistogram ? undefined : series,
          type: isHistogram ? "histogram" : isBar ? "bar" : "scatter",
          mode: isHistogram || isBar ? undefined : isScatter ? "markers" : "lines",
          fill: isArea ? "tozeroy" : undefined,
          marker: { color: "#b24bf3" },
          line: { color: "#ff2a6d", width: 2 },
          histnorm: isHistogram ? "" : undefined,
          nbinsx: isHistogram ? Math.min(30, Math.max(6, Math.round(series.length / 4))) : undefined,
          xbins: isHistogram ? { size: undefined } : undefined,
          ybins: undefined,
          values: isHistogram ? series : undefined,
        },
      ],
      layout: {
        title: chart.title || "",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { l: 40, r: 20, t: 30, b: 40 },
        xaxis: {
          title: chart.x_label || "Step",
          color: "#A0A0B2",
          gridcolor: "rgba(255,255,255,0.06)",
          zerolinecolor: "rgba(255,255,255,0.12)",
        },
        yaxis: {
          title: chart.y_label || "Value",
          color: "#A0A0B2",
          gridcolor: "rgba(255,255,255,0.06)",
          zerolinecolor: "rgba(255,255,255,0.12)",
        },
        font: { color: "#E6E6F0", size: 10 },
      },
      meta: {
        caption: chart.title || "Timeline",
        sourceLabel: "Latest simulation",
      },
    };
  }

  if (chart.source === "latest_run.summary.status_counts" && chart.kind === "pie") {
    const counts = evidence?.latest_run?.summary?.status_counts || {};
    const labels = Object.keys(counts);
    const values = labels.map((k) => Number(counts[k] ?? 0));
    return {
      data: [
        {
          labels,
          values,
          type: "pie",
          marker: { colors: ["#05d5fa", "#ffaa00", "#ff2a6d"] },
          textinfo: "label+percent",
        },
      ],
      layout: {
        title: chart.title || "Status Breakdown",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { l: 10, r: 10, t: 30, b: 10 },
        font: { color: "#E6E6F0", size: 10 },
        showlegend: true,
        legend: { orientation: "h", y: -0.1 },
      },
      meta: {
        caption: chart.title || "Status Breakdown",
        sourceLabel: "Latest simulation",
      },
    };
  }

  if (chart.source === "bank_profiles_sample" && chart.y_field) {
    const banks = evidence?.bank_profiles_sample || [];
    const labels = banks.map((b) => b.name || b.bank_id || "Bank");
    const values = banks.map((b) => Number(b[chart.y_field] ?? 0));
    const kind = ["bar", "scatter", "pie", "histogram"].includes(chart.kind)
      ? chart.kind
      : "bar";
    return {
      data: [
        {
          x: kind === "pie" || kind === "histogram" ? undefined : labels,
          y: kind === "pie" || kind === "histogram" ? undefined : values,
          labels: kind === "pie" ? labels : undefined,
          values: kind === "pie" ? values : undefined,
          type: kind === "pie" ? "pie" : kind === "histogram" ? "histogram" : kind === "scatter" ? "scatter" : "bar",
          mode: kind === "scatter" ? "markers" : undefined,
          marker: { color: "#00e5ff" },
          nbinsx: kind === "histogram" ? Math.min(30, Math.max(6, Math.round(values.length / 4))) : undefined,
        },
      ],
      layout: {
        title: chart.title || "",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { l: 40, r: 20, t: 30, b: 80 },
        xaxis: {
          title: chart.x_label || "Bank",
          color: "#A0A0B2",
          tickangle: -30,
          gridcolor: "rgba(255,255,255,0.06)",
        },
        yaxis: {
          title: chart.y_label || chart.y_field,
          color: "#A0A0B2",
          gridcolor: "rgba(255,255,255,0.06)",
        },
        font: { color: "#E6E6F0", size: 10 },
        showlegend: kind === "pie",
      },
      meta: {
        caption: chart.title || "Bank sample",
        sourceLabel: "Bank Explorer sample",
      },
    };
  }

  return null;
}

function AssistantContent({ content, evidence }) {
  const parsed = tryParseJson(content);
  if (!parsed || typeof parsed !== "object") {
    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        className="prose prose-invert prose-sm max-w-none"
      >
        {normalizeText(content)}
      </ReactMarkdown>
    );
  }

  const title = parsed.title || "Summary";
  const summary = normalizeText(parsed.summary || "");
  const points = Array.isArray(parsed.key_points)
    ? parsed.key_points.map((p) => normalizeText(p))
    : [];
  const limitations = normalizeText(parsed.limitations || "");
  const chartSpec = parsed.chart;
  const chart = buildChartFromEvidence(chartSpec, evidence);
  const confidence = parsed.confidence;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="h-3.5 w-3.5 text-neon-purple" />
          <span className="text-xs font-semibold text-text-primary">{title}</span>
        </div>
        {typeof confidence === "number" && (
          <span className="text-[10px] text-text-muted font-[family-name:var(--font-mono)]">
            {Math.round(confidence * 100)}%
          </span>
        )}
      </div>
      {summary && (
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          className="prose prose-invert prose-sm max-w-none"
        >
          {summary}
        </ReactMarkdown>
      )}
      {chart && (
        <div className="rounded-xl border border-white/10 bg-gradient-to-br from-white/[0.04] to-white/[0.01] p-2">
          <div className="h-[220px] min-h-[200px]">
            <Plot
              data={chart.data}
              layout={chart.layout}
              style={{ width: "100%", height: "100%" }}
              useResizeHandler
              config={{ displayModeBar: false, responsive: true }}
            />
          </div>
          <div className="mt-2 text-[10px] text-text-muted flex items-center justify-between">
            <span>{chart.meta?.caption}</span>
            <span className="text-neon-purple/70">{chart.meta?.sourceLabel}</span>
          </div>
        </div>
      )}
      {points.length > 0 && (
        <ul className="list-disc list-inside text-[11px] text-text-secondary space-y-1">
          {points.map((pt, i) => (
            <li key={i}>
              <ReactMarkdown remarkPlugins={[remarkGfm]} className="prose prose-invert prose-sm max-w-none">
                {pt}
              </ReactMarkdown>
            </li>
          ))}
        </ul>
      )}
      {limitations && (
        <p className="text-[10px] text-text-muted">Limitations: {limitations}</p>
      )}
    </div>
  );
}

export default function Chatbot() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [bankName, setBankName] = useState("");
  const listRef = useRef(null);
  const [panelPos, setPanelPos] = useState({ x: 0, y: 0 });
  const [panelSize, setPanelSize] = useState({ width: 380, height: 520 });

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      if (Array.isArray(saved)) setMessages(saved);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      const savedPos = JSON.parse(localStorage.getItem(POS_KEY) || "null");
      const savedSize = JSON.parse(localStorage.getItem(SIZE_KEY) || "null");
      if (savedPos?.x != null && savedPos?.y != null) setPanelPos(savedPos);
      if (savedSize?.width && savedSize?.height) setPanelSize(savedSize);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-MAX_MESSAGES)));
  }, [messages]);

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages, open]);

  const runContext = useMemo(() => {
    const runId = localStorage.getItem("encs:lastRunId") || null;
    const runType = localStorage.getItem("encs:lastRunType") || null;
    return { runId, runType };
  }, [open]);

  const sendMessage = async () => {
    const content = input.trim();
    if (!content || loading) return;
    const nextMessages = [...messages, { role: "user", content }];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);
    setError(null);
    try {
      const res = await chatWithAssistant({
        messages: nextMessages.map((m) => ({ role: m.role, content: m.content })),
        runId: runContext.runId,
        runType: runContext.runType,
        bankName: bankName || null,
      });
      setMessages((prev) => [...prev, { role: "assistant", content: res.response, evidence: res.evidence }]);
    } catch (e) {
      setError(e.message || "Chat failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 pointer-events-none z-[60]">
      <AnimatePresence>
        {open && (
          <Rnd
            bounds="window"
            size={panelSize}
            position={panelPos}
            minWidth={320}
            minHeight={360}
            maxWidth={720}
            maxHeight={820}
            onDragStop={(_, d) => {
              const next = { x: d.x, y: d.y };
              setPanelPos(next);
              localStorage.setItem(POS_KEY, JSON.stringify(next));
            }}
            onResizeStop={(_, __, ref, ___, pos) => {
              const nextSize = { width: ref.offsetWidth, height: ref.offsetHeight };
              setPanelSize(nextSize);
              setPanelPos(pos);
              localStorage.setItem(SIZE_KEY, JSON.stringify(nextSize));
              localStorage.setItem(POS_KEY, JSON.stringify(pos));
            }}
            dragHandleClassName="chatbot-drag-handle"
            className="pointer-events-auto"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.96, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: 10 }}
              transition={{ duration: 0.18 }}
              className="h-full"
            >
              <GlassPanel
                data-lenis-prevent
                className="h-full flex flex-col p-0 overflow-hidden"
              >
                <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-gradient-to-r from-void-panel/90 via-void-panel/80 to-void-panel/90 chatbot-drag-handle cursor-move">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="h-4 w-4 text-neon-purple" />
                    <span className="text-sm font-bold text-text-primary">ENCS Assistant</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setMessages([])}
                      className="p-1 rounded hover:bg-white/5 text-text-muted hover:text-white"
                      title="Clear"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => setOpen(false)}
                      className="p-1 rounded hover:bg-white/5 text-text-muted hover:text-white"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                <div className="px-4 py-2 border-b border-border text-[10px] text-text-muted flex items-center justify-between bg-white/[0.02]">
                  <span>
                    Context: {runContext.runId ? "Latest simulation loaded" : "No simulation yet"}
                  </span>
                  <span className="text-[10px] text-neon-purple/80 font-[family-name:var(--font-mono)]">
                    {runContext.runType || ""}
                  </span>
                </div>

                <div className="px-4 py-2 border-b border-border bg-white/[0.02]">
                  <label className="text-[10px] text-text-muted uppercase tracking-wider">Bank (optional)</label>
                  <input
                    value={bankName}
                    onChange={(e) => setBankName(e.target.value)}
                    placeholder="Type bank name"
                    className="mt-1 w-full h-8 px-2 rounded bg-white/5 border border-border text-xs text-text-primary"
                  />
                </div>

                <div ref={listRef} className="flex-1 overflow-auto px-4 py-4 space-y-3 scrollbar-thin">
                  {messages.length === 0 && (
                    <div className="text-xs text-text-muted">Ask about simulations, risks, or glossary terms.</div>
                  )}
                  {messages.map((m, i) => (
                    <motion.div
                      key={i}
                      className={cn(
                        "text-xs leading-relaxed p-3 rounded-2xl max-w-[85%] shadow-sm",
                        m.role === "user"
                          ? "bg-neon-purple/15 text-text-primary self-end ml-auto border border-neon-purple/20"
                          : "bg-white/5 text-text-secondary border border-white/10"
                      )}
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.15 }}
                    >
                      {m.role === "assistant" ? (
                        <AssistantContent content={m.content} evidence={m.evidence} />
                      ) : (
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          className="prose prose-invert prose-sm max-w-none"
                        >
                          {normalizeText(m.content)}
                        </ReactMarkdown>
                      )}
                    </motion.div>
                  ))}
                  {loading && <div className="text-xs text-text-muted">Thinking…</div>}
                  {error && <div className="text-xs text-crisis-red">{error}</div>}
                </div>

                <div className="p-3 border-t border-border flex items-center gap-2 bg-void-panel/60">
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                    placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
                    className="flex-1 h-10 px-3 py-2 rounded-lg bg-white/5 border border-border text-xs text-text-primary resize-none"
                    rows={1}
                  />
                  <button
                    onClick={sendMessage}
                    disabled={loading}
                    className="h-9 w-9 flex items-center justify-center rounded-lg bg-neon-purple/20 text-neon-purple hover:bg-neon-purple/30 disabled:opacity-50"
                  >
                    <Send className="h-4 w-4" />
                  </button>
                </div>
              </GlassPanel>
            </motion.div>
          </Rnd>
        )}
      </AnimatePresence>

      {!open && (
        <div className="absolute bottom-4 right-4 pointer-events-auto">
          <button
            onClick={() => setOpen(true)}
            className="h-12 w-12 rounded-full bg-neon-purple/20 text-neon-purple border border-neon-purple/40 flex items-center justify-center shadow-lg"
          >
            <MessageSquare className="h-5 w-5" />
          </button>
        </div>
      )}
    </div>
  );
}
