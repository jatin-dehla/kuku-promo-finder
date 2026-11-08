"""Find promos in a screenplay pack.  Usage (see README.md):

    python run.py --input input/episodes.txt --out outputs/run1
    python run.py --input input/episodes.txt --out outputs/run1 --judge file --judgments judgments.json
    python run.py --input new_show.txt --out outputs/new --judge openai --model gpt-4o-mini

Pipeline:  parse -> generate+score windows -> shortlist -> (LLM judge) -> validate -> rank -> write files
Everything except the optional LLM judge is deterministic Python.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse import load_episodes, read_input_text, fmt  # noqa: E402
from score import (score_episode, pick_top, episode_final_style, overlap, WEIGHTS)  # noqa: E402
from judge import build_prompt, get_judgments  # noqa: E402
from validate import split_source, validate_promo  # noqa: E402

# final = (1-LLM_MIX)*heuristic + LLM_MIX*llm_score   (only when the judgment passes validation)
LLM_MIX = 0.7
LLM_WEIGHTS = (0.35, 0.35, 0.30)   # hook, cliffhanger, standalone


def iou(a, b):
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union else 0.0


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def parse_range(s, all_numbers):
    if not s:
        return set(all_numbers)
    keep = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        keep.update(range(int(a), int(b or a) + 1))
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help=".txt file, folder of .txt, or .zip")
    ap.add_argument("--out", required=True,
                    help="output folder (raw/ and work/ are created inside)")
    ap.add_argument("--judge", default=os.environ.get("PROMO_LLM",
                    "none"), choices=["none", "file", "openai", "anthropic"])
    ap.add_argument("--judgments", default=os.environ.get("PROMO_JUDGMENTS",
                    ""), help="JSON file for --judge file")
    ap.add_argument("--model", default=os.environ.get("PROMO_LLM_MODEL", ""))
    ap.add_argument("--top-k", type=int, default=3,
                    help="structural candidates per episode sent to the judge")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--episodes", default="", help="e.g. 1-5,9 (default: all)")
    ap.add_argument("--redact", action="store_true",
                    help="omit verbatim script quotes from raw outputs")
    args = ap.parse_args()
    os.environ["PROMO_LLM"], os.environ["PROMO_JUDGMENTS"], os.environ["PROMO_LLM_MODEL"] = args.judge, args.judgments, args.model

    t0 = time.time()
    out = Path(args.out)
    raw, work = out / "raw", out / "work"
    raw.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    log = []

    def say(msg):
        print(msg)
        log.append(msg)

    # ---- 1. parse -------------------------------------------------------------------------
    source_text = read_input_text(args.input)
    sha = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    episodes = load_episodes(args.input)
    keep = parse_range(args.episodes, [e.number for e in episodes])
    episodes = [e for e in episodes if e.number in keep]
    say(f"input={args.input} sha256(text)={sha[:16]}.. episodes={len(episodes)} "
        f"events={sum(len(e.events) for e in episodes)} dialogue_lines={sum(len(e.dialogue) for e in episodes)}")
    source = split_source(source_text)

    # ---- 2. candidates + scores -----------------------------------------------------------
    all_rows, summary, shortlist = [], [], []
    by_ep = {e.number: e for e in episodes}
    for ep in episodes:
        cands = score_episode(ep)
        all_rows.extend(cands)
        structural = pick_top(cands, "heuristic_total", args.top_k)
        base_pick = pick_top(cands, "baseline_keyword", 1)
        items = []
        for rank, c in enumerate(structural, 1):
            items.append({**c, "sources": [f"structural#{rank}"]})
        for b in base_pick:
            same = next((it for it in items if iou(it, b) > 0.5), None)
            if same:
                same["sources"].append("keyword_baseline")
            else:
                items.append({**b, "sources": ["keyword_baseline_only"]})
        # shuffle so ids don't reveal the heuristic rank
        rnd = random.Random(args.seed * 1000 + ep.number)
        rnd.shuffle(items)
        for i, it in enumerate(items, 1):
            it["id"] = f"E{ep.number:02d}-{i}"
            shortlist.append(it)
        summary.append({"episode": ep.number, "duration_s": round(ep.duration, 2), "events": len(ep.events),
                        "dialogue_lines": len(ep.dialogue), "valid_windows": len(cands),
                        "final_beat_style": episode_final_style(ep), "source_warnings": len(ep.warnings)})
    say(f"scored {len(all_rows)} valid 20-60s windows; shortlist={len(shortlist)} clips")

    # ---- 3. LLM judge (optional, swappable; see judge.py) ---------------------------------
    prompts, evidence = {}, {}
    for it in shortlist:
        ep = by_ep[it["episode"]]
        prompts[it["id"]] = build_prompt(ep.events, it["start"], it["end"])
        lines = [d for d in ep.dialogue if d.start >=
                 it["start"] - 1e-6 and d.end <= it["end"] + 1e-6]
        evidence[it["id"]] = (lines[0].text, lines[-1].text,
                              lines[0].speaker, lines[-1].speaker)
    with open(work / "judge_prompts.jsonl", "w", encoding="utf-8") as f:   # private: contains clip text
        for cid, p in prompts.items():
            f.write(json.dumps({"id": cid, "prompt": p}) + "\n")
    judgments, status = get_judgments([it["id"] for it in shortlist], prompts)
    say(f"judge: {status}")
    jout = {k: ({**v, "hook_quote": "[redacted]", "cliffhanger_quote": "[redacted]"} if args.redact else v)
            for k, v in judgments.items()}
    json.dump({"status": status, "judgments": jout}, open(
        raw / "llm_judgments.json", "w", encoding="utf-8"), indent=1)

    # ---- 4. validate every shortlisted clip against the raw source ------------------------
    report, ranked = {}, []
    for it in shortlist:
        promo = {"episode": it["episode"],
                 "segments": [(it["start"], it["end"])]}
        j = judgments.get(it["id"])
        if j:
            promo["judgment"] = j
        errs, warns = validate_promo(promo, source)
        report[it["id"]] = {"errors": errs, "warnings": warns}
        structural_ok = not [e for e in errs if not e.startswith(
            ("E_HOOK_QUOTE", "E_CLIFF_QUOTE"))]
        quotes_ok = not [e for e in errs if e.startswith(
            ("E_HOOK_QUOTE", "E_CLIFF_QUOTE"))]
        h = it["heuristic_total"]
        llm_ok = bool(j) and quotes_ok
        llm_score = None
        if llm_ok:
            try:
                llm_score = (LLM_WEIGHTS[0] * float(j["hook_strength"]) + LLM_WEIGHTS[1] * float(j["cliffhanger_strength"])
                             + LLM_WEIGHTS[2] * float(j["standalone"])) / 5.0
            except (KeyError, TypeError, ValueError):
                llm_ok = False
        final = (1 - LLM_MIX) * h + LLM_MIX * llm_score if llm_ok else h
        fl, ll, fs, ls = evidence[it["id"]]
        row = {
            "id": it["id"], "episode": it["episode"], "start": fmt(it["start"]), "end": fmt(it["end"]),
            "duration_s": it["duration"], "segments": [[fmt(it["start"]), fmt(it["end"])]],
            "sources": it["sources"],
            "hook_h": it["hook"], "cliff_h": it["cliff"], "standalone_h": it["standalone"], "pace_h": it["pace"],
            "heuristic_total": h, "baseline_keyword": it["baseline_keyword"],
            "llm_score": round(llm_score, 3) if llm_score is not None else None,
            "final_score": round(final, 4), "rank_basis": "heuristic+llm" if llm_ok else "heuristic_only",
            "verdict": (j or {}).get("verdict") if llm_ok else None,
            "hook": (j or {}).get("hook_quote") if llm_ok and (j or {}).get("hook_quote") else f'{fs.title()}: "{fl}" (first spoken line)',
            "cliffhanger": (j or {}).get("cliffhanger_quote") if llm_ok and (j or {}).get("cliffhanger_quote") else f'{ls.title()}: "{ll}" (last spoken line)',
            "why_it_works": (j or {}).get("why_it_works") if llm_ok else
            f"Heuristic only (no model judgment): hook {it['hook']:.2f}, cliffhanger {it['cliff']:.2f}, standalone {it['standalone']:.2f}, pace {it['pace']:.2f}.",
            "description_source": "llm" if llm_ok else "heuristic_template",
            "llm_hook_strength": j.get("hook_strength") if llm_ok else None,
            "llm_cliff_strength": j.get("cliffhanger_strength") if llm_ok else None,
            "llm_standalone": j.get("standalone") if llm_ok else None,
            "unresolved_question": j.get("unresolved_question") if llm_ok else None,
            "standalone_issue": j.get("standalone_issue") if llm_ok else None,
            "validation_errors": errs, "validation_warnings": warns,
            "valid_structure": structural_ok, "llm_quotes_valid": quotes_ok if j else None,
        }
        ranked.append(row)
    ranked.sort(key=lambda r: (-r["final_score"], r["episode"], r["start"]))
    for i, r in enumerate(ranked, 1):
        r["global_rank"] = i

    # posting queue: best non-skip clip per episode, then ordered by score
    queue = []
    no_promo = []
    for ep in episodes:
        mine = [r for r in ranked if r["episode"] ==
                ep.number and r["valid_structure"] and r["verdict"] != "skip"]
        if mine:
            queue.append(mine[0])
        else:
            no_promo.append(ep.number)
    queue.sort(key=lambda r: -r["final_score"])
    for i, r in enumerate(queue, 1):
        r["queue_position"] = i
    for s in summary:
        s["recommended_clip"] = next(
            (r["id"] for r in queue if r["episode"] == s["episode"]), "NONE")

    # ---- 5. baseline-vs-method comparison (measured on the shortlist only) -----------------
    cmp = {"note": "heuristic-vs-heuristic and heuristic-vs-LLM comparisons on the shortlist; the LLM is a judge, not ground truth"}
    s1 = {}
    for it in shortlist:
        if any(s == "structural#1" for s in it["sources"]):
            s1[it["episode"]] = it
    base = {it["episode"]: it for it in shortlist if any(
        s.startswith("keyword_baseline") for s in it["sources"])}
    same = [n for n in s1 if n in base and iou(s1[n], base[n]) > 0.5]
    cmp["episodes"] = len(s1)
    cmp["baseline_pick_equals_structural_top1"] = len(same)
    by_id = {r["id"]: r for r in ranked}
    if any(r["llm_score"] is not None for r in ranked):
        pairs = [(by_id[s1[n]["id"]]["llm_score"], by_id[base[n]["id"]]["llm_score"]) for n in s1
                 if n in base and n not in same and by_id[s1[n]["id"]]["llm_score"] is not None
                 and by_id[base[n]["id"]]["llm_score"] is not None]
        if pairs:
            cmp["episodes_where_picks_differ_and_both_judged"] = len(pairs)
            cmp["mean_llm_score_structural_top1"] = round(
                sum(a for a, _ in pairs) / len(pairs), 3)
            cmp["mean_llm_score_keyword_baseline"] = round(
                sum(b for _, b in pairs) / len(pairs), 3)
            cmp["structural_top1_beats_baseline_in"] = sum(
                1 for a, b in pairs if a > b)
            cmp["baseline_beats_structural_top1_in"] = sum(
                1 for a, b in pairs if b > a)
            cmp["ties"] = sum(1 for a, b in pairs if a == b)
        agree = 0
        judged_eps = 0
        for n in s1:
            cl = [r for r in ranked if r["episode"] ==
                  n and r["llm_score"] is not None and r["id"] != "x"]
            struct = [r for r in cl if any(
                s.startswith("structural") for s in r["sources"])]
            if len(struct) >= 2:
                judged_eps += 1
                best = max(struct, key=lambda r: r["llm_score"])
                top1 = by_id[s1[n]["id"]]
                agree += best["id"] == top1["id"]
        cmp["llm_best_of_structural_top3_equals_heuristic_top1"] = f"{agree}/{judged_eps} episodes"
        n_err = sum(1 for r in ranked if r["llm_quotes_valid"] is False)
        cmp["llm_judgments_failing_quote_check"] = f"{n_err}/{sum(1 for r in ranked if r['llm_quotes_valid'] is not None)}"
        cmp["llm_judgments_with_name_warnings"] = sum(1 for r in ranked if any(
            w.startswith("W_NAME") for w in r["validation_warnings"]))

    # ---- 6. write raw outputs --------------------------------------------------------------
    if args.redact:
        for r in ranked:
            r["hook"] = r["cliffhanger"] = "[redacted]"
    write_csv(raw / "all_candidates.csv", all_rows)
    write_csv(raw / "episodes_summary.csv", summary)
    flat = [{**r, "segments": json.dumps(r["segments"]), "sources": "|".join(r["sources"]),
             "validation_errors": " || ".join(r["validation_errors"]), "validation_warnings": " || ".join(r["validation_warnings"])}
            for r in ranked]
    write_csv(raw / "shortlist_ranked.csv", flat)
    json.dump({"weights_heuristic": WEIGHTS, "llm_mix": LLM_MIX, "llm_weights_hook_cliff_standalone": LLM_WEIGHTS,
               "posting_queue": queue, "episodes_without_recommendation": no_promo, "all_shortlisted_ranked": ranked},
              open(raw / "promos_ranked.json", "w", encoding="utf-8"), indent=1)
    json.dump(report, open(raw / "validation_report.json",
              "w", encoding="utf-8"), indent=1)
    json.dump(cmp, open(raw / "method_comparison.json",
              "w", encoding="utf-8"), indent=1)
    n_struct_err = sum(1 for r in ranked if not r["valid_structure"])
    say(f"validation: {len(ranked)} clips checked, structural errors={n_struct_err}, "
        f"llm quote failures={sum(1 for r in ranked if r['llm_quotes_valid'] is False)}")
    say(f"posting queue: {len(queue)} episodes with a recommended clip; no recommendation: {no_promo}")
    say(f"seconds={time.time() - t0:.1f} python={platform.python_version()} redact={args.redact} weights={WEIGHTS}")
    (raw / "run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
