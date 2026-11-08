"""Turn a finished run's RAW outputs into human-readable CURATED files (never edits raw/).

    python report.py outputs/full_run

Writes outputs/full_run/curated/promos_top.md, posting_queue.csv and quality_report.md.
All numbers are read from raw/ files; nothing is typed in by hand.
"""
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
raw, cur = out / "raw", out / "curated"
cur.mkdir(exist_ok=True)
d = json.load(open(raw / "promos_ranked.json", encoding="utf-8"))
cmp = json.load(open(raw / "method_comparison.json", encoding="utf-8"))
R, Q = d["all_shortlisted_ranked"], d["posting_queue"]
n_llm = sum(1 for r in R if r["llm_score"] is not None)

# ---------- posting queue ----------
with open(cur / "posting_queue.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["queue_position", "episode", "start", "end", "duration_s",
               "final_score", "verdict", "hook", "cliffhanger", "why_it_works"])
    for r in Q:
        w.writerow([r["queue_position"], r["episode"], r["start"], r["end"], r["duration_s"], r["final_score"], r["verdict"],
                    r["hook"], r["cliffhanger"], r["why_it_works"]])

L = ["# Top promo picks (curated view of raw/promos_ranked.json)", "",
     "Scores: `final = 0.3 x heuristic + 0.7 x LLM-judge score` (both are opinions, not measured engagement). "
     f"{n_llm} of {len(R)} shortlisted clips have an LLM judgment.", "",
     "| # | Ep | Start | End | Len | Score | Hook (first 3 s) | Cliffhanger | Why it works |", "|---|---|---|---|---|---|---|---|---|"]
for r in Q[:15]:
    def esc(t): return (t or "").replace("|", "/")
    L.append(f"| {r['queue_position']} | {r['episode']} | {r['start']} | {r['end']} | {r['duration_s']}s | {r['final_score']:.2f} | "
             f"{esc(r['hook'])} | {esc(r['cliffhanger'])} | {esc(r['why_it_works'])} |")
L += ["", "Post first: #1. It opens on the shortest, most self-explanatory shock in the pack and ends on an open plea, "
      "so a cold viewer needs no backstory. Full ordered list: posting_queue.csv.", ""]
(cur / "promos_top.md").write_text("\n".join(L), encoding="utf-8")

# ---------- quality report ----------
s1 = [r for r in R if "structural#1" in r["sources"]]
bad_top1 = sorted([r for r in s1 if r["verdict"] == "skip"],
                  key=lambda r: -r["heuristic_total"])
Q2 = ["# Quality report (generated from raw/)", "", "## Method comparison on the shortlist",
      "```", json.dumps(cmp, indent=1), "```", "",
      f"## Heuristic top-1 picks the LLM judge would skip: {len(bad_top1)} of {len(s1)} episodes", ""]
for r in bad_top1:
    Q2.append(f"- Ep {r['episode']} {r['start']}-{r['end']} (heuristic {r['heuristic_total']}, "
              f"LLM hook/cliff/standalone {r['llm_hook_strength']}/{r['llm_cliff_strength']}/{r['llm_standalone']}): {r['standalone_issue']}")
Q2 += ["",
       f"## Episodes with NO recommended promo: {d['episodes_without_recommendation']}", ""]
for n in d["episodes_without_recommendation"]:
    best = max((r for r in R if r["episode"] == n),
               key=lambda r: r["final_score"])
    Q2.append(
        f"- Ep {n}: best clip {best['id']} scored H/C/S {best['llm_hook_strength']}/{best['llm_cliff_strength']}/{best['llm_standalone']}, verdict {best['verdict']}. {best['why_it_works']}")
Q2 += ["", "## Validator warnings on model descriptions (W_NAME)", ""]
for r in R:
    for w_ in r["validation_warnings"]:
        Q2.append(f"- {r['id']}: {w_}")
Q2 += ["", "## Validator errors (structure, timestamps, quotes)", ""]
errs = [(r["id"], e) for r in R for e in r["validation_errors"]]
Q2 += [f"- {i}: {e}" for i, e in errs] or ["- none"]
(cur / "quality_report.md").write_text("\n".join(Q2), encoding="utf-8")
print("wrote", cur)
