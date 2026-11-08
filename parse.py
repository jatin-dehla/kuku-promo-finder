"""Parse Kuku-style timestamped screenplays into plain Python objects.

Everything here is deterministic. No AI is involved.

Input can be:
  * one .txt file holding many episodes (each starting with "EPISODE NUMBER: ..."),
  * a folder of .txt files, or
  * a .zip of .txt files.

Block format found in the real data (checked on all 50 episodes):

    **CAMERA:**                          <- header: kind or "SPEAKER (emotion)"
    Start: (00:09.80) | End: (00:11.00)  <- times as (MM:SS.ss)
    text ...                             <- dialogue is wrapped in double quotes
"""
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

HEADER_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
TIME_RE = re.compile(r"Start:\s*\((\d+):(\d+(?:\.\d+)?)\)\s*\|\s*End:\s*\((\d+):(\d+(?:\.\d+)?)\)")
EPISODE_RE = re.compile(r"^EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", re.I)
SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
NON_DIALOGUE = ("CAMERA", "ACTION", "SOUND FX")


@dataclass
class Event:
    kind: str          # CAMERA, ACTION, SOUND FX or DIALOGUE
    speaker: str       # only for DIALOGUE
    emotion: str       # text of the last (...) in the header, e.g. "Cold, Callous"
    start: float       # seconds
    end: float
    text: str
    scene: int


@dataclass
class Episode:
    number: int
    logline: str = ""
    continuity: str = ""
    events: list = field(default_factory=list)
    scenes: list = field(default_factory=list)      # (scene_no, heading, start, end)
    warnings: list = field(default_factory=list)

    @property
    def duration(self):
        return max((e.end for e in self.events), default=0.0)

    @property
    def dialogue(self):
        return [e for e in self.events if e.kind == "DIALOGUE"]


def to_seconds(minutes, seconds):
    return int(minutes) * 60 + float(seconds)


def fmt(t):
    """Seconds -> 'MM:SS.ss' (same shape as the source files)."""
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:05.2f}"


def parse_ts(s):
    """'MM:SS.ss' -> seconds."""
    m, sec = s.split(":")
    return to_seconds(m, sec)


def read_input_text(path):
    """Return the whole input as one string (file, folder of .txt, or .zip)."""
    p = Path(path)
    if p.is_dir():
        return "\n".join(f.read_text(encoding="utf-8") for f in sorted(p.rglob("*.txt")))
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(".txt") and "__MACOSX" not in n)
            return "\n".join(z.read(n).decode("utf-8") for n in names)
    return p.read_text(encoding="utf-8")


def split_episodes(text):
    """Cut the text at every 'EPISODE NUMBER:' line."""
    lines = text.split("\n")
    starts = [i for i, l in enumerate(lines) if EPISODE_RE.match(l)]
    chunks = []
    for a, b in zip(starts, starts[1:] + [len(lines)]):
        chunks.append(lines[a:b])
    return chunks


def parse_episode(lines):
    ep = Episode(number=int(EPISODE_RE.match(lines[0]).group(1)))
    scene_no, scene_heading, scene_times = 0, "", [None, None]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        low = line.lower()
        if low.startswith("episode logline:"):
            ep.logline = line.split(":", 1)[1].strip()
        elif low.startswith("continuity notes:"):
            ep.continuity = line.split(":", 1)[1].strip()
        elif SCENE_RE.match(line):
            scene_no = int(SCENE_RE.match(line).group(1))
            scene_heading, scene_times = "", [None, None]
            ep.scenes.append([scene_no, "", None, None])
        elif low.startswith("scene heading:") and ep.scenes:
            ep.scenes[-1][1] = line.split(":", 1)[1].strip()
        elif low.startswith("scene start:") and ep.scenes:
            m, s = line.split(":", 1)[1].strip().split(":")
            ep.scenes[-1][2] = to_seconds(m, s)
        elif low.startswith("scene end:") and ep.scenes:
            m, s = line.split(":", 1)[1].strip().split(":")
            ep.scenes[-1][3] = to_seconds(m, s)
        else:
            h = HEADER_RE.match(line)
            if h and i + 1 < len(lines):
                t = TIME_RE.match(lines[i + 1].strip())
                if t:
                    start = to_seconds(t.group(1), t.group(2))
                    end = to_seconds(t.group(3), t.group(4))
                    # body = following lines up to the next blank line
                    j, body = i + 2, []
                    while j < len(lines) and lines[j].strip():
                        body.append(lines[j].strip())
                        j += 1
                    text = " ".join(body)
                    header = h.group(1)
                    base = header.split(" (")[0].strip()
                    parens = re.findall(r"\(([^)]*)\)", header)
                    emotion = parens[-1] if parens else ""
                    if base.upper() in NON_DIALOGUE:
                        kind, speaker = base.upper(), ""
                    else:
                        kind, speaker = "DIALOGUE", base.upper()
                        text = text.strip().strip('"').strip()
                    ep.events.append(Event(kind, speaker, emotion, start, end, text, scene_no))
                    i = j
                    continue
        i += 1
    ep.events.sort(key=lambda e: (e.start, e.end))
    check_episode(ep)
    return ep


def check_episode(ep):
    """Data-quality warnings about the SOURCE (not about our output)."""
    if not ep.events:
        ep.warnings.append("no events parsed")
    for e in ep.events:
        if e.end < e.start:
            ep.warnings.append(f"{e.kind} ends before it starts at {fmt(e.start)}")
    prev = None
    for d in ep.dialogue:
        if prev and d.start < prev.end - 0.3:
            ep.warnings.append(f"dialogue overlap {fmt(prev.start)}-{fmt(prev.end)} vs {fmt(d.start)}")
        prev = d
    for no, _, s, e in ep.scenes:
        if s is None or e is None:
            ep.warnings.append(f"scene {no} missing start/end")


def load_episodes(path):
    chunks = split_episodes(read_input_text(path))
    episodes = [parse_episode(c) for c in chunks]
    numbers = [e.number for e in episodes]
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"duplicate episode numbers: {sorted(n for n in set(numbers) if numbers.count(n) > 1)}")
    return sorted(episodes, key=lambda e: e.number)


if __name__ == "__main__":
    import sys
    eps = load_episodes(sys.argv[1])
    print(f"{len(eps)} episodes, numbers {eps[0].number}..{eps[-1].number}")
    for e in eps:
        print(f"Ep {e.number:2d}: {len(e.events):3d} events, {len(e.dialogue):3d} dialogue, "
              f"duration {fmt(e.duration)}, scenes {len(e.scenes)}, warnings {len(e.warnings)}")
