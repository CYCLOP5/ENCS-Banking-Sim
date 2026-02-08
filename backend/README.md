---
title: ENCS Systemic Risk Engine
emoji: 🏦
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 7860
# Force rebuild 1
---

# ENCS Systemic Risk Engine (Backend)

This is the backend for the ENCS Systemic Risk Simulation, hosted on Hugging Face Spaces.

## API Endpoints

- `POST /api/simulate`: Run standard Eisenberg-Noe contagion simulation.
- `POST /api/game`: Run strategic bank run simulation (Morris & Shin).
- `POST /api/climate`: Run Green Swan climate risk simulation.
- `POST /api/explain/run`: Generate LLM-based explanation for a simulation run.
- `POST /api/explain/bank`: Generate LLM-based analysis for a specific bank.

## Environment Variables

Ensure these secrets are set in your Space settings:

- `GROQ_API_KEY`: API key for Groq (LLM provider).
- `GROQ_MODEL`: Model ID (e.g., `llama3-70b-8192`).
