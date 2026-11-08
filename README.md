# Promo Finder (Kuku AI Generalist assignment, Problem 1)

Reads timestamped microdrama screenplays and proposes 20-60 s promo clips: episode, start/end timestamps,
hook, cliffhanger, one-line "why it works", a ranking, and an automatic check of every claim against the source.

**Confidential data is not in this repo.** Put the screenplay pack in input/ yourself (it is git-ignored).

## Quick start (no API key required)

Run the prototype using deterministic heuristic ranking:

```bash
python run.py --input input/episodes.txt --out outputs/run1
python report.py outputs/run1
python tests.py input/episodes.txt
```

This mode requires no LLM or API key. It uses heuristic scores and produces ranked promo candidates, but does not perform semantic AI judging.

## Reproducing the AI-assisted results

To reproduce the original AI-assisted results for the supplied 50-episode screenplay pack, place the privately supplied `llm_judgments.json` alongside `run.py` and execute:

```bash
python run.py --input input/episodes.txt --out outputs/ai_final --judge file --judgments llm_judgments.json
python report.py outputs/ai_final
python tests.py input/episodes.txt
```

**Verified local results:**
- 50 episodes processed
- 34,823 valid candidate windows
- 138 shortlisted clips
- 47 episodes recommended
- Episodes 6, 14 and 17 excluded
- Zero structural validation errors
- All 21 automated tests passed

This mode reuses previously generated model judgments; it does not make new LLM calls. For new screenplay inputs, use the heuristic mode or generate fresh judgments through a configured model provider. Live API integrations have only been tested against mock servers, not production services.

`--input` can be one `.txt` with many episodes, a folder of `.txt` files, or a `.zip` of them.
Each episode must start with `EPISODE NUMBER: ...` and use blocks like
`**CAMERA:**` / `**SPEAKER (emotion):**` followed by `Start: (MM:SS.ss) | End: (MM:SS.ss)`.

Without an LLM the ranking uses the heuristic score only, and the "hook / cliffhanger" text is the first / last spoken line
of the clip, labelled `heuristic_template`. That is a baseline, not semantic understanding.

## Adding a language model (the swap point is one function)
judge.py: call_llm(prompt) -> str. Select the provider using --judge and --model, with API credentials supplied through environment variables.

| Provider | How |
|---|---|
| any OpenAI-compatible API (OpenAI, Groq, Together, Ollama, Gemini compat URL...) | `python run.py ... --judge openai --model MODEL` with `PROMO_LLM_API_KEY`, optional `PROMO_LLM_BASE_URL` |
| Anthropic | `--judge anthropic --model MODEL` with `PROMO_LLM_API_KEY` |
| free, no API (works with any chat model) | run once, open `outputs/X/work/judge_prompts.jsonl`, paste each prompt into any chat model, save the JSON answers as `{"E01-1": {...}, ...}`, then `--judge file --judgments answers.json` |

Only the shortlist (3 structural picks + 1 keyword pick per episode, about 138 clips for 50 episodes) is sent to the model.
Clip ids are shuffled per episode so they do not reveal the heuristic rank. Runs are deterministic (`--seed`).
**Tested status:** `file` and `none` were used on the real data. `openai` and `anthropic` were tested only against a local mock
server (see `tests.py`), never against the real services, because the build environment had no network.

## What runs where
| Step | Code | Notes |
|---|---|---|
| parse screenplay | `parse.py` | deterministic |
| all valid 20-60 s windows | `score.py` | never cuts a dialogue line; a spoken line must start within 3 s of the start |
| heuristic score = 0.35 hook + 0.30 cliffhanger + 0.20 standalone + 0.15 pacing | `score.py` | rules over text, screenplay emotion tags and camera/action text. A proxy, not measured engagement |
| keyword-only baseline | `score.py` | deliberately naive, for comparison |
| semantic judging (hook, cliffhanger, standalone, why-it-works) | `judge.py` | optional, swappable |
| validation against the raw source | `validate.py` | independent of the parser; checks timestamps, 20-60 s, order, cut lines, 3 s hook slot, verbatim quotes, unknown names |
| final score = 0.3 heuristic + 0.7 LLM (only when the judgment passes the quote check) | `run.py` | |

## Outputs (`outputs/<run>/raw/`, produced by the program, never edited)
`all_candidates.csv` (every valid window, no text), `shortlist_ranked.csv`, `promos_ranked.json` (posting queue, per-clip
scores and validation), `llm_judgments.json`, `validation_report.json`, `method_comparison.json`, `episodes_summary.csv`, `run_log.txt`.
`report.py` writes a separate `curated/` view. `work/` holds the prompts (full clip text) and is not meant for sharing.

## Confidentiality
`raw/` contains short verbatim quotes (the hook and cliffhanger lines). `--redact` removes them; the redacted folder was checked
for verbatim dialogue lines and none were found. Do not publish `input/`, `work/` or non-redacted `raw/` in a public repo.

## Known limits
No human labels and no engagement data, so scores are opinions (heuristic and model) and are not conversion predictions.
Single continuous segments only (multi-segment promos are not built). No video work. No UI.
