"""The ONLY place where an LLM is touched. Swap the model by changing environment variables.

    PROMO_LLM=none        no LLM; pipeline ranks by heuristic score only (default)
    PROMO_LLM=file        read pre-made judgments from a JSON file (free fallback: paste the
                          exported prompts into ANY chat model, save its answers, point to the file)
    PROMO_LLM=openai      any OpenAI-compatible chat endpoint (OpenAI, Groq, Together, Ollama,
                          Gemini's OpenAI-compatible URL, a local llama.cpp server, ...)
    PROMO_LLM=anthropic   Anthropic Messages API

    PROMO_LLM_MODEL, PROMO_LLM_BASE_URL, PROMO_LLM_API_KEY, PROMO_JUDGMENTS (for file)

Only the standard library is used, so there is nothing to install.
"""
import json
import os
import re
import urllib.request

from parse import fmt

RUBRIC = """You are a short-form video editor choosing promo clips (TikTok / Reels / ReelShort style)
from a screenplay of an AI-generated microdrama. A promo is a 20-60 second continuous clip shown to
people who have NEVER seen the show.

Judge ONLY the clip below. Do not use outside knowledge of the show.

Scores are integers 1-5:
 hook_strength: does the FIRST 3 SECONDS stop a scrolling viewer? (5 = shocking reveal / slap / brutal line;
   3 = interesting but ordinary; 1 = flat or needs context to matter)
 cliffhanger_strength: does the clip END on an unresolved moment that makes the viewer need the next part?
   (5 = open threat / reveal / question left hanging; 3 = tension but partly resolved; 1 = scene resolves or fades out)
 standalone: can a new viewer follow who is who and what is at stake? (5 = fully clear; 1 = lost without prior episodes)

Rules:
 - hook_quote must be copied EXACTLY from a dialogue line that STARTS within the first 3 seconds (or "" if the hook is purely visual).
 - cliffhanger_quote must be copied EXACTLY from one of the last 3 dialogue lines, or from the final ACTION text.
 - Do not mention any person, object or event that is not in the clip text.
 - why_it_works is ONE sentence.
 - verdict: "post" (I would publish this), "maybe", or "skip".

Reply with ONLY a JSON object with these keys:
hook_quote, hook_strength, cliffhanger_quote, cliffhanger_strength, unresolved_question,
standalone, standalone_issue, why_it_works, verdict
"""


def clip_transcript(events, start, end):
    """Plain-text view of the clip: dialogue (with emotion tag), ACTION and SOUND FX. Camera text is skipped."""
    out = []
    for e in events:
        if e.end < start - 1e-6 or e.start > end + 1e-6:
            continue
        if e.kind == "DIALOGUE" and e.start >= start - 1e-6 and e.end <= end + 1e-6:
            tag = f" ({e.emotion})" if e.emotion else ""
            out.append(f'[{fmt(e.start - start)}] {e.speaker}{tag}: "{e.text}"')
        elif e.kind in ("ACTION", "SOUND FX") and e.start >= start - 1e-6 and e.start <= end:
            out.append(f"[{fmt(e.start - start)}] {e.kind}: {e.text[:160]}")
    return "\n".join(out)


def build_prompt(events, start, end):
    return RUBRIC + f"\nCLIP (length {end - start:.1f}s, times are relative to the clip start):\n" + clip_transcript(events, start, end)


# ----------------------------------------------------------------------------------------
# THE ONE FUNCTION TO SWAP
# ----------------------------------------------------------------------------------------
def call_llm(prompt):
    """prompt (str) -> reply (str). Raises RuntimeError if the provider is not usable."""
    provider = os.environ.get("PROMO_LLM", "none").lower()
    model = os.environ.get("PROMO_LLM_MODEL", "")
    key = os.environ.get("PROMO_LLM_API_KEY", "")
    if provider == "openai":
        base = os.environ.get("PROMO_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        body = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["choices"][0]["message"]["content"]
    if provider == "anthropic":
        base = os.environ.get("PROMO_LLM_BASE_URL", "https://api.anthropic.com").rstrip("/")
        body = {"model": model, "max_tokens": 800, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(base + "/v1/messages", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "x-api-key": key,
                                      "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["content"][0]["text"]
    raise RuntimeError(f"PROMO_LLM={provider!r} cannot answer prompts (use openai or anthropic)")


def parse_reply(text):
    """Pull the first JSON object out of a model reply (models sometimes wrap it in prose/code fences)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    return json.loads(m.group(0))


def get_judgments(clips, prompts):
    """clips: list of clip ids; prompts: {id: prompt}. Returns ({id: judgment}, status_text)."""
    provider = os.environ.get("PROMO_LLM", "none").lower()
    if provider == "none":
        return {}, "no LLM (PROMO_LLM=none): ranking uses heuristic score only"
    if provider == "file":
        path = os.environ.get("PROMO_JUDGMENTS", "")
        if not path or not os.path.exists(path):
            return {}, f"judgments file not found ({path!r}): ranking uses heuristic score only"
        data = json.load(open(path, encoding="utf-8"))
        data = data.get("judgments", data)
        return {k: v for k, v in data.items() if k in prompts}, f"loaded judgments from {path}"
    out, errors = {}, []
    for cid in clips:
        try:
            out[cid] = parse_reply(call_llm(prompts[cid]))
        except Exception as ex:  # keep going; a missing judgment just means heuristic-only for that clip
            errors.append(f"{cid}: {type(ex).__name__}: {ex}")
    return out, f"{provider} model={os.environ.get('PROMO_LLM_MODEL')!r}: {len(out)}/{len(clips)} ok; errors={errors[:5]}"
