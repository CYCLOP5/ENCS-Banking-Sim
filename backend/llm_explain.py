"""llm_explain.py — Summarization and Groq LLM integration."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import httpx


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _downsample(series: list, max_points: int = 60) -> list:
    if not isinstance(series, list) or len(series) <= max_points:
        return series
    step = max(1, len(series) // max_points)
    return [series[i] for i in range(0, len(series), step)]


def _series_stats(series: list) -> Dict[str, Any]:
    if not series:
        return {"min": None, "max": None, "last": None, "peak_step": None}
    min_val = min(series)
    max_val = max(series)
    peak_step = int(series.index(max_val)) + 1
    return {"min": float(min_val), "max": float(max_val), "last": float(series[-1]), "peak_step": peak_step}


def build_graph_evidence(results: Dict[str, Any]) -> Dict[str, Any]:
    series_map = {
        "price_timeline": results.get("price_timeline") or [],
        "defaults_timeline": results.get("defaults_timeline") or [],
        "distressed_timeline": results.get("distressed_timeline") or [],
        "gridlock_timeline": results.get("gridlock_timeline") or [],
        "margin_calls_timeline": results.get("margin_calls_timeline") or [],
        "equity_loss_timeline": results.get("equity_loss_timeline") or [],
    }
    return {
        "series": {k: _downsample(v) for k, v in series_map.items()},
        "stats": {k: _series_stats(v) for k, v in series_map.items()},
        "lengths": {k: len(v) for k, v in series_map.items()},
    }


def build_run_summary(run_type: str, results: Dict[str, Any]) -> Dict[str, Any]:
    if run_type in {"mechanical", "climate"}:
        status = results.get("status") or []
        status_counts = {
            "Safe": int(sum(1 for s in status if s == "Safe")),
            "Distressed": int(sum(1 for s in status if s == "Distressed")),
            "Default": int(sum(1 for s in status if s == "Default")),
        }
        return {
            "run_type": run_type,
            "n_defaults": int(results.get("n_defaults", 0) or 0),
            "n_distressed": int(results.get("n_distressed", 0) or 0),
            "equity_loss": float(results.get("equity_loss", 0) or 0),
            "final_asset_price": float(results.get("final_asset_price", 1) or 1),
            "trigger_name": results.get("trigger_name"),
            "loss_severity": float(results.get("loss_severity", 0) or 0),
            "graphs": build_graph_evidence(results),
            "status_counts": status_counts,
        }
    if run_type == "game":
        opaque = results.get("opaque", {})
        transparent = results.get("transparent", {})
        return {
            "run_type": run_type,
            "capital_saved": float(results.get("capital_saved", 0) or 0),
            "opaque_run_rate": float(opaque.get("run_rate", 0) or 0),
            "transparent_run_rate": float(transparent.get("run_rate", 0) or 0),
        }
    return {"run_type": run_type}


def _extract_bank_impact(results: Dict[str, Any], bank_name: str) -> Optional[Dict[str, Any]]:
    names = results.get("bank_names") or []
    status = results.get("status") or []
    initial = results.get("initial_equity") or []
    final = results.get("final_equity") or []
    for i, name in enumerate(names):
        if str(name).strip() == str(bank_name).strip():
            return {
                "status": status[i] if i < len(status) else None,
                "initial_equity": float(initial[i]) if i < len(initial) else None,
                "final_equity": float(final[i]) if i < len(final) else None,
                "equity_loss": float(initial[i]) - float(final[i]) if i < len(initial) and i < len(final) else None,
            }
    return None


def build_bank_context(
    bank_profile: Dict[str, Any],
    run_type: Optional[str],
    run_results: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    context = {"bank_profile": bank_profile}
    if run_results and run_type in {"mechanical", "climate"}:
        impact = _extract_bank_impact(run_results, bank_profile.get("name", ""))
        if impact:
            context["bank_impact"] = impact
            context["run_type"] = run_type
    return context


def build_prompt(
    query: str,
    evidence: Dict[str, Any],
) -> str:
    return (
        "You are an analyst summarizing systemic risk simulation outputs for non-expert users. "
        "Use only the provided evidence. If information is missing, say you do not know. "
        "Return a JSON object with keys: title, summary, key_points (array), limitations, confidence (0-1).\n\n"
        "CRITICAL FORMATTING RULES — follow these exactly:\n"
        "1. All mathematical expressions MUST be wrapped in single dollar signs for inline math: $P_{t+1} = P_t \\cdot e^{-\\alpha V_t / 10^{12}}$.\n"
        "2. Use LaTeX commands for Greek letters: $\\alpha$, $\\beta$, $\\sigma$, etc.\n"
        "3. Never write bare LaTeX outside dollar signs. Every subscript, superscript, or formula must be inside $...$.\n"
        "4. For words in subscripts/superscripts, use \\text{...} with no spaces, e.g., $P_{\\text{sell}}$.\n"
        "5. Never duplicate an equation — write it exactly once.\n"
        "6. Never use Unicode Greek letters (α, β) or Unicode superscripts (¹²³) — always use $\\alpha$, $10^{12}$, etc.\n"
        "7. Do not start a line with a bare colon.\n\n"
        f"User question: {query}\n\n"
        f"Evidence JSON: {json.dumps(evidence)}"
    )


def build_bank_prompt(query: str, evidence: Dict[str, Any]) -> str:
    return (
        "You are an analyst summarizing a single bank using the provided evidence only. "
        "Return a JSON object with keys: title, summary, key_points (array), limitations, confidence (0-1). "
        "Prefer concrete numbers from evidence (assets, equity, leverage, risk score, percentiles). "
        "If peer comparisons are available, mention them briefly. If data is missing, say you do not know.\n\n"
        "CRITICAL FORMATTING RULES — follow these exactly:\n"
        "1. All mathematical expressions MUST be wrapped in single dollar signs for inline math: $P_{t+1} = P_t \\cdot e^{-\\alpha V_t / 10^{12}}$.\n"
        "2. Use LaTeX commands for Greek letters: $\\alpha$, $\\beta$, $\\sigma$, etc.\n"
        "3. Never write bare LaTeX outside dollar signs.\n"
        "4. For words in subscripts/superscripts, use \\text{...} with no spaces, e.g., $P_{\\text{sell}}$.\n"
        "5. Never duplicate an equation — write it exactly once.\n"
        "6. Never use Unicode Greek letters or superscripts — always use LaTeX inside $...$.\n\n"
        f"User question: {query}\n\n"
        f"Evidence JSON: {json.dumps(evidence)}"
    )


def call_groq(prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    model_name = model or os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a careful financial risk assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(GROQ_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    content = data["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"title": "Summary", "summary": content, "key_points": [], "limitations": "", "confidence": 0.4}


def load_site_knowledge(path: Optional[str] = None) -> Dict[str, Any]:
    file_path = path or os.environ.get(
        "ENCS_SITE_KNOWLEDGE_PATH",
        str(os.path.join(os.path.dirname(__file__), "site_knowledge.json")),
    )
    base = {"site": "ENCS Systemic Risk Engine", "pages": []}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            base = json.load(f)
    except Exception:
        base = {"site": "ENCS Systemic Risk Engine", "pages": []}

    frontend_root = os.environ.get(
        "ENCS_FRONTEND_PATH",
        str(Path(__file__).resolve().parents[1] / "frontend" / "src" / "pages"),
    )
    pages = []
    try:
        for file_name in [
            "Landing.jsx",
            "Methodology.jsx",
            "Implementation.jsx",
            "Simulation.jsx",
            "Terminology.jsx",
            "BankExplorer.jsx",
        ]:
            fp = Path(frontend_root) / file_name
            if not fp.exists():
                continue
            text = fp.read_text(encoding="utf-8", errors="ignore")
            extracted = _extract_text_from_jsx(text)
            pages.append({
                "id": file_name.replace(".jsx", ""),
                "source": str(fp),
                "content": extracted[:8000],
            })
    except Exception:
        pages = []

    base["pages_raw"] = pages
    return base


def _extract_text_from_jsx(text: str) -> str:
    text = re.sub(r"^import .*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def call_groq_chat(messages: list[dict], evidence: Dict[str, Any]) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    model_name = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
    system_prompt = (
        "You are the ENCS assistant. Answer clearly for non-expert users. "
        "Use only the provided evidence. If information is missing, say you do not know. "
        "Keep responses concise and factual. "
        "Return a JSON object with keys: title, summary, key_points (array), limitations, confidence (0-1). "
        "If the user asks for a chart, add a 'chart' object with: "
        "{kind: 'line'|'bar'|'pie'|'scatter'|'area'|'histogram', title, source, series_key, x_label, y_label, y_field}. "
        "Choose a feasible chart type based on available evidence; if the requested type is not feasible, pick the closest alternative. "
        "Use source 'latest_run.graphs.series' with series_key one of: "
        "price_timeline, defaults_timeline, distressed_timeline, gridlock_timeline, margin_calls_timeline, equity_loss_timeline. "
        "For status pies, use source 'latest_run.summary.status_counts'. "
        "Or use source 'bank_profiles_sample' with y_field like gnn_risk_score or total_assets.\n\n"
        "CRITICAL FORMATTING RULES — follow these exactly:\n"
        "1. All mathematical expressions MUST be wrapped in single dollar signs for inline math: $P_{t+1} = P_t \\cdot e^{-\\alpha V_t / 10^{12}}$.\n"
        "2. Use LaTeX commands for Greek letters: $\\alpha$, $\\beta$, $\\sigma$, etc.\n"
        "3. Never write bare LaTeX outside dollar signs. Every subscript, superscript, or formula must be inside $...$.\n"
        "4. For words in subscripts/superscripts, use \\text{...} with no spaces, e.g., $P_{\\text{sell}}$.\n"
        "5. Never duplicate an equation — write it exactly once.\n"
        "6. Never use Unicode Greek letters (α, β) or Unicode superscripts (¹²³) — always use $\\alpha$, $10^{12}$, etc.\n"
        "7. Do not start a line with a bare colon.\n\n"
        f"Evidence JSON: {json.dumps(evidence)}"
    )
    payload = {
        "model": model_name,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(GROQ_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"]
