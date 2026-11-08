"""Run:  python tests.py input/episodes.txt
Plain asserts, no test framework. Prints PASS/FAIL per check."""
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse import load_episodes, read_input_text, parse_ts  # noqa: E402
from score import score_episode  # noqa: E402
from validate import split_source, validate_promo  # noqa: E402
import judge  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "input/episodes.txt"
results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


eps = load_episodes(SRC)
source = split_source(read_input_text(SRC))

# ---- parser -------------------------------------------------------------------------------
check("parser: 50 episodes numbered 1..50", [e.number for e in eps] == list(range(1, 51)))
check("parser: 3727 events (matches independent grep count of **HEADER:** blocks)", sum(len(e.events) for e in eps) == 3727)
check("parser: Ep1 first line is the matriarch at 11.3 s", eps[0].dialogue[0].start == 11.3 and eps[0].dialogue[0].text.startswith("You should turn around"))
check("parser: Ep50 present with a final cut-to-black description", "cut to black" in eps[-1].events[-1].text.lower())

# ---- every generated window passes the INDEPENDENT validator ---------------------------------
n = bad = 0
for ep in eps:
    for c in score_episode(ep):
        n += 1
        errs, _ = validate_promo({"episode": ep.number, "segments": [(c["start"], c["end"])]}, source)
        bad += bool(errs)
        if errs and bad <= 3:
            print("   e.g.", ep.number, c["start"], c["end"], errs[:2])
check(f"generator vs independent validator: {n} windows, 0 errors", bad == 0, f"{bad} bad")

# ---- negative tests: the validator must catch deliberately wrong promos -------------------------
def codes(promo):
    e, w = validate_promo(promo, source)
    return {x.split(":")[0] for x in e + w}


check("neg: 70 s clip -> E_DURATION", "E_DURATION" in codes({"episode": 1, "segments": [(11.3, 81.3)]}))
check("neg: invented timestamp -> E_TS_SOURCE", "E_TS_SOURCE" in codes({"episode": 1, "segments": [(11.37, 41.2)]}))
check("neg: cut inside a dialogue line -> E_STRADDLE", "E_STRADDLE" in codes({"episode": 1, "segments": [(12.0, 45.0)]}))
check("neg: no dialogue within first 3 s -> E_HOOK_WINDOW", "E_HOOK_WINDOW" in codes({"episode": 1, "segments": [(0.0, 35.0)]}))
check("neg: unknown episode -> E_EPISODE", "E_EPISODE" in codes({"episode": 99, "segments": [(1, 30)]}))
check("neg: segments out of order -> E_ORDER", "E_ORDER" in codes({"episode": 1, "segments": [(41.7, 60.0), (11.3, 25.8)]}))
good = {"episode": 1, "segments": [(11.3, 41.2)]}
check("pos: Ep1 11.30-41.20 is clean", not validate_promo(good, source)[0], str(validate_promo(good, source)[0]))
j = {"hook_quote": "You should turn around, Juliana.", "cliffhanger_quote": "Because he is marrying me.",
     "why_it_works": "Juliana learns Gerald is marrying her sister.", "unresolved_question": "", "standalone_issue": ""}
check("pos: correct verbatim quotes pass", not validate_promo({**good, "judgment": j}, source)[0], str(validate_promo({**good, "judgment": j}, source)))
bad_j = {**j, "hook_quote": "You will pay for this, Juliana!"}
check("neg: invented hook quote -> E_HOOK_QUOTE", "E_HOOK_QUOTE" in codes({**good, "judgment": bad_j}))
bad_j = {**j, "cliffhanger_quote": "Gerald is a traitor."}
check("neg: invented cliffhanger quote -> E_CLIFF_QUOTE", "E_CLIFF_QUOTE" in codes({**good, "judgment": bad_j}))
bad_j = {**j, "why_it_works": "Bartholomew interrupts the wedding."}
check("neg: invented character name -> W_NAME", "W_NAME" in codes({**good, "judgment": bad_j}))

# ---- swappable LLM layer, tested against a local mock server (NOT a real model) -----------------
class Mock(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        reply = 'Sure! ```json\n{"hook_strength": 4, "verdict": "post"}\n```'
        body = ({"choices": [{"message": {"content": reply}}]} if self.path.endswith("/chat/completions")
                else {"content": [{"type": "text", "text": reply}]})
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


srv = HTTPServer(("127.0.0.1", 0), Mock)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}"
for prov in ("openai", "anthropic"):
    os.environ.update(PROMO_LLM=prov, PROMO_LLM_BASE_URL=base, PROMO_LLM_MODEL="mock", PROMO_LLM_API_KEY="x")
    got, status = judge.get_judgments(["E01-1"], {"E01-1": "prompt"})
    check(f"judge provider '{prov}' (mock server): request, fenced-JSON reply parsed", got.get("E01-1", {}).get("verdict") == "post", status[:60])
srv.shutdown()
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "j.json"
    p.write_text(json.dumps({"judgments": {"E01-1": {"verdict": "maybe"}, "E99-9": {"verdict": "post"}}}))
    os.environ.update(PROMO_LLM="file", PROMO_JUDGMENTS=str(p))
    got, _ = judge.get_judgments(["E01-1"], {"E01-1": "x"})
    check("judge provider 'file': loads only ids that exist", list(got) == ["E01-1"])
os.environ["PROMO_LLM"] = "none"
got, status = judge.get_judgments(["E01-1"], {"E01-1": "x"})
check("judge provider 'none': returns nothing, pipeline falls back to heuristic", got == {})

# ---- determinism: two runs give byte-identical ranked output --------------------------------------
hashes = []
for _ in range(2):
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([sys.executable, "-I", str(Path(__file__).parent / "run.py"), "--input", SRC, "--out", td, "--episodes", "1-12"],
                       check=True, capture_output=True)
        hashes.append(hashlib.sha256((Path(td) / "raw" / "shortlist_ranked.csv").read_bytes()).hexdigest())
check("determinism: identical shortlist_ranked.csv across two runs", hashes[0] == hashes[1])

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
