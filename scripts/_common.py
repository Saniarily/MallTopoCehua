"""Shared CLI helpers for scripts."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[], help="dotted overrides key=value")
    p.add_argument("--log-level", default="INFO")
    return p
