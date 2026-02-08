---
title: ENCS Systemic Risk Engine
emoji: 🏦
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ENCS Systemic Risk Engine API

A FastAPI backend for the Eisenberg-Noe Contagion Simulation engine.

## Endpoints

- `GET /api/health` - Health check
- `GET /api/topology` - Network graph data
- `GET /api/banks` - Bank data with risk scores
- `POST /api/simulate` - Run contagion simulation
- `POST /api/climate` - Run climate shock simulation
- `POST /api/game` - Run game theory simulation
