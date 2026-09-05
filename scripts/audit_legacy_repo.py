#!/usr/bin/env python3
"""Read-only audit of the legacy MallTopoRanker repository -> docs/legacy_audit_generated.md + JSON."""
from __future__ import annotations
import argparse, json
from _common import ROOT
from mall_space_planner.data.audit import audit_legacy_repo, render_markdown
from mall_space_planner.utils import setup_logging

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--legacy-repo", required=True); p.add_argument("--out-dir", default=str(ROOT / "outputs/reports"))
    a = p.parse_args(); setup_logging("INFO")
    rep = audit_legacy_repo(a.legacy_repo)
    out = ROOT / a.out_dir if not a.out_dir.startswith("/") else __import__("pathlib").Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "legacy_repo_audit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "legacy_repo_audit.md").write_text(render_markdown("Legacy repository audit (generated)", rep), encoding="utf-8")
    print(f"Wrote {out / 'legacy_repo_audit.md'}")

if __name__ == "__main__":
    main()
