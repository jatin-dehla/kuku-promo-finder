"""Independent, deterministic validation of promo candidates against the RAW source text.

This file deliberately does NOT import parse.py. It re-reads the source with its own tiny regex so a
bug in the parser cannot silently validate itself.

Checks per promo (each failure is returned as an ERROR string; WARN strings are softer):
  E_EPISODE      episode number exists in the source
  E_TS_SOURCE    start/end timestamps are literally present in that episode's text
  E_ORDER        segments in playback order, non-overlapping, end > start
  E_DURATION     total duration 20-60 s
  E_STRADDLE     a boundary cuts through a dialogue line
  E_HOOK_WINDOW  first dialogue line starts within 3 s of the clip start
  E_HOOK_QUOTE   LLM hook quote is not verbatim text from a line that starts in the first 3 s
  E_CLIFF_QUOTE  LLM cliffhanger quote is not verbatim from the last 3 lines / final action
  W_NAME         a capitalised name in the LLM free text does not occur anywhere in the clip text
"""
import re

BLOCK_RE = re.compile(
    r"\*\*(.+?):\*\*\s*\nStart: \((\d+):(\d+(?:\.\d+)?)\) \| End: \((\d+):(\d+(?:\.\d+)?)\)\s*\n(.*)")
TS_RE = re.compile(r"\((\d+):(\d+\.\d+)\)")
MIN_DUR, MAX_DUR, HOOK_WINDOW = 20.0, 60.0, 3.0
ORDINARY_SUFFIX = re.compile(r"(s|ing|ly|ed|tion|ship|ity|ness|ty|ment|ous|ive|al)$")
_VOCAB = {}
STOP_CAPS = {"the", "a", "an", "he", "she", "it", "they", "this", "that", "his", "her", "their", "i", "we",
             "you", "what", "who", "why", "how", "will", "can", "does", "is", "are", "was", "were", "do",
             "if", "but", "and", "or", "no", "yes", "not", "viewer", "viewers", "new", "clip", "episode",
             "post", "maybe", "skip", "none", "scene", "s", "my", "our", "your", "its", "before", "after",
             "when", "while", "with", "without", "from", "to", "of", "in", "on", "at", "for", "by", "as"}


def norm(s):
    s = s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("…", "...").replace("—", "-").replace("–", "-")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def secs(m, s):
    return int(m) * 60 + float(s)


def split_source(text):
    """{episode_number: raw_text_of_episode}"""
    parts = re.split(r"(?m)^(?=EPISODE NUMBER:)", text)
    out = {}
    for p in parts:
        m = re.match(r"EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", p)
        if m:
            out[int(m.group(1))] = p
    return out


def blocks(raw):
    res = []
    for m in BLOCK_RE.finditer(raw):
        head = m.group(1)
        kind = head.split(" (")[0].strip().upper()
        res.append({
            "kind": kind if kind in ("CAMERA", "ACTION", "SOUND FX") else "DIALOGUE",
            "start": secs(m.group(2), m.group(3)), "end": secs(m.group(4), m.group(5)),
            "text": m.group(6).strip().strip('"'),
            "start_lit": f"({m.group(2)}:{m.group(3)})", "end_lit": f"({m.group(4)}:{m.group(5)})",
        })
    return res


def fmt_lit(t):
    m = int(t // 60)
    return f"({m:02d}:{t - 60 * m:05.2f})"


def validate_promo(promo, source):
    """promo = {'episode': int, 'segments': [(start, end), ...], optional 'judgment': {...}}.
    Returns (errors, warnings)."""
    errs, warns = [], []
    ep = promo["episode"]
    if ep not in source:
        return [f"E_EPISODE: episode {ep} not in source"], warns
    bl = blocks(source[ep])
    dlg = sorted((b for b in bl if b["kind"] == "DIALOGUE"), key=lambda b: b["start"])
    starts_lit = {b["start_lit"] for b in bl}
    ends_lit = {b["end_lit"] for b in bl}
    prev_end = -1.0
    total = 0.0
    for (s, e) in promo["segments"]:
        if fmt_lit(s) not in starts_lit and not any(abs(b["start"] - s) < 0.006 for b in bl):
            errs.append(f"E_TS_SOURCE: start {fmt_lit(s)} not found as a Start: in episode {ep}")
        if fmt_lit(e) not in ends_lit and not any(abs(b["end"] - e) < 0.006 for b in bl):
            errs.append(f"E_TS_SOURCE: end {fmt_lit(e)} not found as an End: in episode {ep}")
        if e <= s or s < prev_end:
            errs.append(f"E_ORDER: segment {fmt_lit(s)}-{fmt_lit(e)} not in order / non-positive")
        prev_end = e
        total += e - s
        for t, name in ((s, "start"), (e, "end")):
            for d in dlg:
                if d["start"] + 0.05 < t < d["end"] - 0.05:
                    errs.append(f"E_STRADDLE: {name} {fmt_lit(t)} cuts dialogue {fmt_lit(d['start'])}-{fmt_lit(d['end'])}")
    total = round(total, 2)
    if not (MIN_DUR <= total <= MAX_DUR):
        errs.append(f"E_DURATION: total {total}s outside {MIN_DUR:.0f}-{MAX_DUR:.0f}s")

    s0, e0 = promo["segments"][0][0], promo["segments"][0][1]
    first = [d for d in dlg if d["start"] >= s0 - 1e-6]
    if not first or first[0]["start"] - s0 > HOOK_WINDOW + 1e-6:
        errs.append("E_HOOK_WINDOW: no dialogue starts within 3 s of the clip start")

    if id(source) not in _VOCAB:   # every lower-case word in the whole pack
        _VOCAB[id(source)] = set(norm(" ".join(source.values())).split())
    vocab = _VOCAB[id(source)]
    j = promo.get("judgment")
    if j:
        clip_dlg = [d for d in dlg if d["start"] >= s0 - 1e-6 and d["end"] <= e0 + 1e-6]
        hq = norm(j.get("hook_quote", "") or "")
        if hq:
            early = [norm(d["text"]) for d in clip_dlg if d["start"] - s0 <= HOOK_WINDOW + 1e-6]
            if not any(hq in t for t in early):
                errs.append(f"E_HOOK_QUOTE: {j.get('hook_quote')!r} is not verbatim in a line starting in the first 3 s")
        cq = norm(j.get("cliffhanger_quote", "") or "")
        if cq:
            tail = [norm(d["text"]) for d in clip_dlg[-3:]]
            tail += [norm(b["text"]) for b in bl if b["kind"] == "ACTION" and s0 <= b["start"] <= e0
                     and b["end"] >= e0 - 4.0]
            if not any(cq in t for t in tail):
                errs.append(f"E_CLIFF_QUOTE: {j.get('cliffhanger_quote')!r} is not verbatim in the last 3 lines / final action")
        corpus = norm(" ".join(b["text"] for b in bl if s0 - 0.01 <= b["start"] <= e0 + 0.01)
                      + " " + " ".join(h for h in re.findall(r"\*\*(.+?):\*\*", source[ep])))
        free = " ".join(str(j.get(k, "")) for k in ("unresolved_question", "standalone_issue", "why_it_works"))
        free_nq = re.sub(r'"[^"]*"', " ", free)
        for sent in re.split(r"(?<=[.!?])\s+", free_nq):
            toks = re.findall(r"[A-Za-z']+", sent)
            for i, tok in enumerate(toks):
                if not tok[0].isupper():
                    continue
                base = tok.lower().rstrip("'s").rstrip("'")
                if base in STOP_CAPS or len(base) < 3:
                    continue
                if i == 0 and (base in vocab or ORDINARY_SUFFIX.search(base)):
                    continue   # sentence-initial ordinary word ("Opens", "Relationships"), not a name
                if base not in corpus:
                    warns.append(f"W_NAME: {tok!r} in the model's description does not appear in the clip text")
    return errs, warns
