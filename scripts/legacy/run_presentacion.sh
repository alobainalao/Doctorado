#!/usr/bin/env bash
# Wrapper mínimo — delega todo a run_presentacion.py
cd "$(dirname "$0")/.."
conda run -n bfr_env python scripts/run_presentacion.py "$@"
