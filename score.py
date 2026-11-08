"""Generate 20-60 s promo windows and score them with transparent, rule-based features.

IMPORTANT HONESTY NOTE
These scores are HEURISTICS computed from script text and the screenplay's own
annotations (emotion tags, camera/action descriptions). They are proxies for
"hook / cliffhanger / standalone clarity". They are NOT measured viewer
engagement and they do not understand story meaning. The optional LLM judge
(judge.py) is the semantic layer; this file only proposes and pre-ranks windows.

A window is one continuous stretch [start, end] of one episode.
Hard rules enforced here (and re-checked independently in validate.py):
  * 20 <= duration <= 60 seconds
  * start and end never cut through a dialogue line
  * first dialogue line starts within 3 s of the window start (the "hook" slot)
  * at least MIN_LINES dialogue lines inside
"""
import re

MIN_DUR, MAX_DUR = 20.0, 60.0
HOOK_WINDOW = 3.0
MIN_LINES = 4
MAX_TRAILING = 6.0   # silence/visual after the last spoken line, unless it is the episode's last beat

# Weights of the final heuristic score. Printed into every output file.
WEIGHTS = {"hook": 0.35, "cliff": 0.30, "standalone": 0.20, "pace": 0.15}

# ---------- small, readable lexicons (all lower-case) ----------
HIGH_STAKES = {
    "dead", "death", "die", "dying", "kill", "killed", "murder", "poison", "poisoned", "arsenic",
    "blood", "witch", "witchcraft", "gallows", "hang", "hangs", "prisoner", "prison", "traitor",
    "betrayed", "betrayal", "secret", "truth", "proof", "confess", "disowned", "revoked", "stole",
    "stolen", "thief", "arrest", "arrested", "impossible", "alive", "vengeance", "destroy",
    "planted", "paid", "fire", "burn", "burning", "duke", "duchess", "magistrate", "married",
    "marry", "husband", "wife", "wedding", "lie", "lied", "liar", "dare", "threat", "ruin",
}
INTENSE_EMOTION = {  # tag word -> arousal weight
    "shouting": 1, "screaming": 1, "scream": 1, "furious": 1, "enraged": 1, "outraged": 1,
    "menacing": 1, "threatening": 1, "venomous": 1, "deadly": 1, "lethal": 1, "roaring": 1,
    "hysterical": 1, "snarl": 1, "terrified": 1, "panicked": 1, "horrified": 1, "shrieking": 1,
    "vicious": .7, "cold": .6, "accusing": .7, "accusatory": .7, "desperate": .7, "pleading": .6,
    "gasping": .6, "shocked": .7, "defiant": .6, "ominous": .7, "dark": .6, "stunned": .6,
    "sinister": .7, "commanding": .6, "harsh": .6, "gloating": .5, "mocking": .5, "cruel": .6,
    "sobbing": .5, "alarmed": .6, "livid": 1, "incredulous": .5, "betrayed": .6, "dangerous": .8,
}
ACTION_DRAMA = (
    "slap", "strike", "struck", "slam", "shove", "seize", "grab", "drag", "hurl", "fling",
    "smash", "crash", "collapse", "kick", "punch", "lunge", "charge", "burst", "snatch", "tear",
    "rip", "whip", "gasp", "reveal", "discover", "withdraw", "pulls", "draws", "knife", "blood",
    "gun", "sword", "faint", "scream", "freeze", "shatter", "throws",
)
RELATION_WORDS = {
    "husband", "wife", "mother", "father", "sister", "brother", "daughter", "son", "duke",
    "duchess", "lord", "lady", "married", "marry", "wedding", "bride", "groom", "family", "child",
    "grace", "mother's", "father's",
}
SOFT_ENDING = (
    "thank you", "i accept", "together", "i love", "goodnight", "good night", "enough for tonight",
    "i trust", "peace", "forever", "sleep", "i will wait", "begin again", "not afraid", "don't feel afraid",
    "i am sorry", "i'm sorry", "well done", "very well", "we'll see", "safe",
)
FILLER_OPENERS = re.compile(r"^(and|but|then|so|or|yes|no|well|because|which|that)\b", re.I)
CONTINUATION = re.compile(r"^(\.\.\.|…|[a-z])")
# Style of the episode's last beat, from the screenplay's own words
END_CLIFF = re.compile(r"cliffhanger|cut to black|abrupt|freeze|hold(s|ing)? on|tension peaks|suspense|"
                       r"ominous|dread|heart-stopping|menacing|hangs|lethal|horrif|terror|shock", re.I)
END_SOFT = re.compile(r"peaceful|warm(ly)?|tender|serenity|radiant|contented|content\b|gentle|"
                      r"fades? (out|into)|triumphant|swelling score|hope", re.I)

WORD_RE = re.compile(r"[A-Za-z']+")


def words(text):
    return [w.lower() for w in WORD_RE.findall(text)]


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


# ---------- per-line features ----------
def emotion_weight(tag):
    ws = words(tag)
    return max((INTENSE_EMOTION.get(w, 0.15) for w in ws), default=0.0)


def stakes_hits(text):
    return sum(1 for w in words(text) if w in HIGH_STAKES)


def opener_score(line, ep_events, window_start):
    """How strong would this dialogue line be as the first thing a scrolling viewer sees (0..1)?"""
    n = len(words(line.text))
    if n <= 2:
        punch = 0.45 if stakes_hits(line.text) == 0 else 0.7
    elif n <= 9:
        punch = 1.0
    else:
        punch = clamp(1.0 - (n - 9) / 25.0, 0.3, 1.0)
    intensity = emotion_weight(line.emotion)
    content = clamp(stakes_hits(line.text) / 2.0)
    t = line.text
    punct = 0.5 * ("?" in t or "!" in t) + 0.5 * bool(re.search(r"\byou(r)?\b", t, re.I))
    # visual hook: dramatic ACTION/CAMERA text starting in the first 3 s
    act = 0.0
    for e in ep_events:
        if e.kind in ("ACTION", "CAMERA") and window_start <= e.start <= window_start + HOOK_WINDOW:
            if any(k in e.text.lower() for k in ACTION_DRAMA):
                act = 1.0
                break
    return 0.30 * punch + 0.25 * intensity + 0.20 * content + 0.15 * punct + 0.10 * act


def closer_score(line, prev_line, end_time, ep, final_style):
    """How unresolved/gripping is it to cut right after this line (0..1)?"""
    t = line.text.strip()
    q = 1.0 if t.endswith("?") else 0.8 if t.endswith(("...", "…", "—", "-")) else 0.0
    threat = clamp(stakes_hits(t) / 2.0)
    inten = emotion_weight(line.emotion)
    at_end = end_time >= ep.duration - 1.0
    if at_end:
        marker = {"cliff": 1.0, "neutral": 0.4, "soft": 0.0}[final_style]
    else:
        marker = 0.5          # mid-episode cut: story visibly continues after it
    last_two = (t + " " + (prev_line.text if prev_line else "")).lower()
    soft = any(k in last_two for k in SOFT_ENDING)
    score = 0.25 * q + 0.40 * marker + 0.15 * threat + 0.20 * inten
    if soft:
        score -= 0.25
    return clamp(score)


def episode_final_style(ep):
    """Classify the episode's last ~4 s of description as cliffhanger / soft / neutral."""
    tail = " ".join(e.text for e in ep.events if e.end >= ep.duration - 4.0 and e.kind != "DIALOGUE")
    tail_all = tail + " " + " ".join(e.text for e in ep.dialogue[-1:])
    if END_CLIFF.search(tail) and not (END_SOFT.search(tail) and not re.search(r"cliffhanger|cut to black|abrupt|freeze", tail, re.I)):
        return "cliff"
    if END_SOFT.search(tail_all):
        return "soft"
    return "neutral"


# ---------- window generation ----------
def straddles(dialogue, t):
    return any(d.start + 0.05 < t < d.end - 0.05 for d in dialogue)


def candidate_windows(ep):
    """Yield (start, end) pairs that satisfy the hard rules."""
    dlg = ep.dialogue
    starts = set()
    for d in dlg:
        starts.add(round(d.start, 2))
        for e in ep.events:
            if e.kind in ("CAMERA", "ACTION") and d.start - 2.0 <= e.start <= d.start:
                starts.add(round(e.start, 2))
    ends = set(round(d.end, 2) for d in dlg)
    ends.add(round(ep.duration, 2))
    for e in ep.events:
        if e.kind in ("ACTION", "CAMERA") and any(0 <= e.end - d.end <= 3.0 for d in dlg):
            ends.add(round(e.end, 2))
    starts = sorted(s for s in starts if not straddles(dlg, s))
    ends = sorted(x for x in ends if not straddles(dlg, x))
    for s in starts:
        inside_after = [d for d in dlg if d.start >= s - 1e-6]
        if not inside_after or inside_after[0].start - s > HOOK_WINDOW:
            continue
        for e in ends:
            dur = round(e - s, 2)
            if dur < MIN_DUR:
                continue
            if dur > MAX_DUR:
                break
            lines = [d for d in dlg if d.start >= s - 1e-6 and d.end <= e + 1e-6]
            if len(lines) < MIN_LINES:
                continue
            trailing = e - lines[-1].end
            if trailing > MAX_TRAILING and e < ep.duration - 0.01:
                continue
            yield s, e, lines


def score_window(ep, s, e, lines, final_style):
    dur = e - s
    first, last = lines[0], lines[-1]
    prev = lines[-2] if len(lines) > 1 else None
    hook = opener_score(first, ep.events, s)
    cliff = closer_score(last, prev, e, ep, final_style)

    # standalone clarity proxies
    speakers = {l.speaker for l in lines}
    n_sp = len(speakers)
    sp = 1.0 if 2 <= n_sp <= 3 else 0.6 if n_sp in (1, 4) else 0.3
    rel = clamp(sum(1 for l in lines if set(words(l.text)) & RELATION_WORDS) / 3.0)
    scene_open = 0.0
    scene_bounds = [sc[2] for sc in ep.scenes if sc[2] is not None]
    if any(abs(s - b) < 0.6 for b in scene_bounds):
        scene_open = 1.0
    elif any(ev.kind == "CAMERA" and "establishing" in ev.text.lower() and abs(ev.start - s) < 0.1 for ev in ep.events):
        scene_open = 1.0
    ft = first.text.strip()
    start_ok = 0.0 if (CONTINUATION.match(ft) or FILLER_OPENERS.match(ft) or len(words(ft)) <= 1) else 1.0
    pron = sum(1 for l in lines[:2] if re.search(r"\b(he|she|him|her|they|them)\b", l.text, re.I)
               and not (set(words(l.text)) & RELATION_WORDS))
    standalone = clamp(0.35 * sp + 0.25 * rel + 0.15 * scene_open + 0.25 * start_ok - 0.1 * pron)

    # pacing: dialogue density + duration sweet spot (30-50 s)
    spoken = sum(l.end - l.start for l in lines)
    density = clamp(spoken / dur / 0.5)
    max_gap = max((b.start - a.end for a, b in zip(lines, lines[1:])), default=0.0)
    gap_pen = clamp((max_gap - 4.0) / 6.0) * 0.5
    if dur < 30:
        shape = 0.5 + 0.5 * (dur - 20) / 10
    elif dur <= 50:
        shape = 1.0
    else:
        shape = 1.0 - 0.4 * (dur - 50) / 10
    pace = clamp(0.6 * density + 0.4 * shape - gap_pen)

    total = (WEIGHTS["hook"] * hook + WEIGHTS["cliff"] * cliff +
             WEIGHTS["standalone"] * standalone + WEIGHTS["pace"] * pace)

    # keyword-only BASELINE (deliberately naive, for comparison)
    base = (3 * stakes_hits(first.text) + 3 * stakes_hits(last.text)
            + sum(l.text.count("?") + l.text.count("!") for l in lines))
    return {
        "episode": ep.number, "start": s, "end": e, "duration": round(dur, 2),
        "n_lines": len(lines), "n_speakers": n_sp, "first_line_start": first.start,
        "hook": round(hook, 3), "cliff": round(cliff, 3), "standalone": round(standalone, 3),
        "pace": round(pace, 3), "heuristic_total": round(total, 4), "baseline_keyword": base,
        "ends_at_episode_end": e >= ep.duration - 0.01, "episode_end_style": final_style,
    }


def score_episode(ep):
    style = episode_final_style(ep)
    return [score_window(ep, s, e, lines, style) for s, e, lines in candidate_windows(ep)]


def overlap(a, b):
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return inter / min(a["duration"], b["duration"])


def pick_top(cands, key, k=3, max_overlap=0.2):
    """Greedy top-k by `key`, rejecting windows that overlap an already chosen one."""
    chosen = []
    for c in sorted(cands, key=lambda c: (-c[key], -c["heuristic_total"], c["start"])):
        if all(overlap(c, x) <= max_overlap for x in chosen):
            chosen.append(c)
        if len(chosen) == k:
            break
    return chosen
