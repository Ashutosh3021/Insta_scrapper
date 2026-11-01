#!/usr/bin/env python3
"""
video_to_text.py  -  links in a .txt file  ->  spoken words + on-screen text in one .txt file.
No LLM. Everything runs on your own computer.

Uses:
  yt-dlp       downloads the video (Instagram, YouTube, TikTok, X ...)
  ffmpeg       pulls out the audio and takes frames
  whisper-cli  turns the audio into text with timestamps (Hindi / English / auto-detect)
  tesseract    reads the text that appears inside the frames (OCR)

Usage:
  python video_to_text.py links.txt -o transcripts.txt

links.txt = one link per line (blank lines and lines starting with # are ignored).
A path to a local video file also works instead of a link.

Each reel gets a live progress bar, e.g.
  [██████████████░░░░░░░░░░░░░░]  50%  transcribing 63%  (0:41)

One-time setup (on top of the guide's yt-dlp / ffmpeg / whisper-cpp):
  Mac:      brew install tesseract tesseract-lang
  Windows:  install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
            (tick "Hindi" under additional language data)
  Linux:    sudo apt install tesseract-ocr tesseract-ocr-hin
  Model:    use the MULTILINGUAL whisper model (ggml-small.bin or ggml-base.bin,
            NOT the ".en" ones) in ~/.cache/whisper/
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

TS_RE = re.compile(
    r"\[(\d+):(\d+):(\d+)[.,](\d+)\s*-->\s*(\d+):(\d+):(\d+)[.,](\d+)\]\s*(.*)"
)
WHISPER_PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+)\s*%")
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
NOISE_WORDS = {
    "music", "blank_audio", "silence", "applause", "laughter", "noise",
    "inaudible", "singing", "speaking in foreign language", "no speech",
}


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def find_tool(names, extra=()):
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    for p in extra:
        if os.path.exists(p):
            return p
    return None


def run(cmd, env=None):
    return subprocess.run(
        [str(c) for c in cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def stream(cmd, watch, on_line):
    """
    Run a command and call on_line(line) for every line of the watched stream
    ("stdout" or "stderr") while it is running. The other stream is collected in
    a background thread so nothing can block.
    Returns (returncode, watched_text, other_text).
    """
    p = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    watched = p.stdout if watch == "stdout" else p.stderr
    other = p.stderr if watch == "stdout" else p.stdout
    box = []
    t = threading.Thread(target=lambda: box.append(other.read()), daemon=True)
    t.start()
    seen = []
    try:
        for line in watched:
            seen.append(line)
            on_line(line)
        p.wait()
        t.join()
    finally:
        if p.poll() is None:  # Ctrl+C or error: don't leave the tool running
            p.kill()
    return p.returncode, "".join(seen), (box[0] if box else "")


def fmt_time(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# live progress bar (one per reel)
# ----------------------------------------------------------------------------
def pick_glyphs():
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "█░".encode(enc)
        return "█", "░"
    except Exception:
        return "#", "-"


class Bar:
    """
    Overall progress is a weighted mix of the stages. Stages that don't apply to a
    reel (no audio, OCR off, local file) are dropped and the rest is re-scaled.
    The weights are rough: they make the bar move steadily, not tell the exact time left.
    """

    LABELS = {
        "download": "downloading",
        "audio": "extracting audio",
        "speech": "transcribing",
        "frames": "grabbing frames",
        "ocr": "reading on-screen text",
    }

    def __init__(self, weights):
        self.w = dict(weights)
        self.f = {k: 0.0 for k in weights}
        self.stage = None
        self.detail = ""
        self.shown = 0.0
        self.t0 = time.monotonic()
        self.last_draw = 0.0
        self.tty = sys.stdout.isatty()
        self.full, self.empty = pick_glyphs()
        self.width = 28
        self.prev_len = 0
        self.last_step = -1
        self.last_stage = None

    def drop(self, name):
        self.w.pop(name, None)
        self.f.pop(name, None)

    def _overall(self):
        total = sum(self.w.values())
        if total <= 0:
            return 1.0
        return sum(self.w[k] * self.f[k] for k in self.w) / total

    def update(self, name, frac, detail=""):
        if name not in self.w:
            return
        self.f[name] = max(self.f[name], min(1.0, max(0.0, frac)))
        self.stage = name
        self.detail = detail
        self.shown = max(self.shown, self._overall())
        self._draw()

    def _line(self, text):
        pct = int(self.shown * 100)
        filled = int(self.width * self.shown)
        bar = self.full * filled + self.empty * (self.width - filled)
        elapsed = fmt_time(time.monotonic() - self.t0)
        return f"  [{bar}] {pct:3d}%  {text}  ({elapsed})"

    def _stage_text(self):
        label = self.LABELS.get(self.stage, "working")
        if self.detail:
            return f"{label} {self.detail}"
        if self.stage in self.f:
            return f"{label} {int(self.f[self.stage] * 100)}%"
        return label

    def _draw(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_draw < 0.1:
            return
        self.last_draw = now
        line = self._line(self._stage_text())
        if self.tty:
            cols = shutil.get_terminal_size((100, 20)).columns - 1
            line = line[:cols]
            sys.stdout.write("\r" + line.ljust(self.prev_len))
            sys.stdout.flush()
            self.prev_len = len(line)
        else:
            # output is redirected to a file: print a plain line every 10%
            step = int(self.shown * 100) // 10
            if step != self.last_step or self.stage != self.last_stage:
                print(line, flush=True)
                self.last_step, self.last_stage = step, self.stage

    def finish(self, ok):
        if ok:
            self.shown = 1.0
        line = self._line("done" if ok else "failed")
        if self.tty:
            cols = shutil.get_terminal_size((100, 20)).columns - 1
            line = line[:cols]
            sys.stdout.write("\r" + line.ljust(self.prev_len) + "\n")
            sys.stdout.flush()
        else:
            print(line, flush=True)


# ----------------------------------------------------------------------------
# step 1: get the video
# ----------------------------------------------------------------------------
def get_video(source, workdir, yt_dlp, cookies_browser, on_progress):
    if os.path.exists(source):
        return Path(source)

    out = workdir / "video.%(ext)s"
    cmd = [
        yt_dlp, "--no-playlist", "--no-warnings",
        "--newline", "--progress", "--progress-template", "download:DLPROG %(progress._percent_str)s",
        "-f", "bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format", "mp4",
        "-o", str(out),
    ]
    if cookies_browser:
        cmd += ["--cookies-from-browser", cookies_browser]
    cmd.append(source)

    def on_line(line):
        m = PERCENT_RE.search(line)
        if m:
            on_progress(float(m.group(1)) / 100)

    rc, _, err = stream(cmd, "stdout", on_line)
    files = [p for p in workdir.glob("video.*") if p.suffix not in (".part", ".ytdl")]
    if rc != 0 or not files:
        tail = err.strip().splitlines()[-3:]
        raise RuntimeError(
            "download failed: " + " | ".join(tail)
            + "  (if it needs a login, re-run with --cookies-from-browser chrome)"
        )
    on_progress(1.0)
    return files[0]


def probe_duration(video):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", video])
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def has_audio(video):
    r = run(["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", video])
    return "audio" in r.stdout


def ffmpeg_run(args, duration, on_progress):
    """Run ffmpeg and report progress (0..1) using its -progress output."""
    cmd = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error",
           "-progress", "pipe:1", "-nostats"] + [str(a) for a in args]

    def on_line(line):
        key, _, val = line.strip().partition("=")
        if key in ("out_time_us", "out_time_ms") and duration > 0:
            try:
                secs = max(0, int(val)) / 1e6
            except ValueError:
                return
            on_progress(min(1.0, secs / duration))

    rc, _, err = stream(cmd, "stdout", on_line)
    if rc == 0:
        on_progress(1.0)
    return rc, err


# ----------------------------------------------------------------------------
# step 2: spoken words (whisper)
# ----------------------------------------------------------------------------
def extract_audio(video, wav, duration, on_progress):
    rc, _ = ffmpeg_run(["-i", video, "-vn", "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", wav], duration, on_progress)
    return rc == 0 and Path(wav).exists()


def is_silent(wav, threshold_db=-50.0):
    r = run(["ffmpeg", "-i", wav, "-af", "volumedetect", "-f", "null", "-"])
    m = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", r.stderr)
    return bool(m) and float(m.group(1)) < threshold_db


def is_noise(text):
    stripped = re.sub(r"[\[\]\(\)\*♪_\s\.\-]+", " ", text).strip().lower()
    if not stripped:
        return True
    return stripped.replace(" ", "_") in {w.replace(" ", "_") for w in NOISE_WORDS} \
        or stripped in NOISE_WORDS


def transcribe(wav, whisper, model, lang, threads, on_progress):
    base = [whisper, "-m", model, "-f", wav, "-l", lang, "-t", threads]

    def on_line(line):
        m = WHISPER_PROGRESS_RE.search(line)
        if m:
            on_progress(int(m.group(1)) / 100)

    # -pp makes whisper-cli print its progress (to stderr) while it works
    rc, err, out = stream(base + ["-pp"], "stderr", on_line)
    if rc != 0 and re.search(r"unknown argument|invalid argument", err, re.I):
        rc, err, out = stream(base, "stderr", lambda line: None)  # older build: no live progress
    if rc != 0:
        raise RuntimeError("whisper failed: " + err.strip()[-300:])
    on_progress(1.0)

    detected = re.search(r"auto-detected language:\s*(\w+)", err)
    segments = []
    for line in out.splitlines():
        m = TS_RE.match(line.strip())
        if not m:
            continue
        h1, m1, s1, _, h2, m2, s2, _, text = m.groups()
        text = text.strip()
        if is_noise(text):
            continue
        start = int(h1) * 3600 + int(m1) * 60 + int(s1)
        end = int(h2) * 3600 + int(m2) * 60 + int(s2)
        segments.append((start, end, text))
    return segments, (detected.group(1) if detected else lang)


# ----------------------------------------------------------------------------
# step 3: on-screen text (frames + OCR)
# ----------------------------------------------------------------------------
def pick_interval(duration, requested):
    if requested:
        return requested
    if duration <= 180:
        return 1
    if duration <= 1200:
        return 5
    return 10


def extract_frames(video, frames_dir, interval, duration, on_progress):
    frames_dir.mkdir(exist_ok=True)
    rc, err = ffmpeg_run(["-i", video, "-vf", f"fps=1/{interval}", "-q:v", "2",
                          str(frames_dir / "f_%05d.jpg")], duration, on_progress)
    if rc != 0:
        raise RuntimeError("frame extraction failed: " + err.strip()[-200:])
    return sorted(frames_dir.glob("f_*.jpg"))


def clean_ocr_line(line):
    line = " ".join(line.split())
    squashed = line.replace(" ", "")
    if len(squashed) < 3:
        return None
    good = sum(1 for ch in squashed if unicodedata.category(ch)[0] in "LMN")
    if good < 3 or good / len(squashed) < 0.6:
        return None
    return line


def ocr_frame(args):
    tesseract, img, langs, psm = args
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    r = run([tesseract, img, "stdout", "-l", langs, "--psm", psm], env=env)
    lines = []
    for raw in r.stdout.splitlines():
        c = clean_ocr_line(raw)
        if c:
            lines.append(c)
    return lines


def normalize(s):
    return re.sub(r"\W+", "", s.lower())


def merge_ocr(per_frame, interval):
    """per_frame = [(time, [lines])]. Merges the same text across frames into one span."""
    entries = []
    gap = interval * 2 + 0.5  # tolerate one missed frame
    for t, lines in per_frame:
        active = [e for e in entries if t - e["last"] <= gap]
        for line in lines:
            norm = normalize(line)
            if not norm:
                continue
            match = None
            for e in active:
                if SequenceMatcher(None, norm, e["norm"]).ratio() >= 0.8:
                    match = e
                    break
            if match:
                match["last"] = t
                if len(line) > len(match["text"]):
                    match["text"] = line
            else:
                e = {"text": line, "norm": norm, "start": t, "last": t}
                entries.append(e)
                active.append(e)
    entries.sort(key=lambda e: (e["start"], e["last"]))
    return entries


def ocr_video(video, workdir, tesseract, langs, interval, psm, workers,
              duration, on_frames, on_ocr):
    frames = extract_frames(video, workdir / "frames", interval, duration, on_frames)
    if not frames:
        on_ocr(1, 1)
        return []
    jobs = [(tesseract, f, langs, psm) for f in frames]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for lines in pool.map(ocr_frame, jobs):
            results.append(lines)
            on_ocr(len(results), len(jobs))
    per_frame = [(i * interval, lines) for i, lines in enumerate(results)]
    return merge_ocr(per_frame, interval)


# ----------------------------------------------------------------------------
# per-link pipeline
# ----------------------------------------------------------------------------
RATE_LIMIT_RE = re.compile(r"rate.?limit|too many requests|HTTP Error 429", re.I)


def get_video_with_retry(source, workdir, yt_dlp, a, bar):
    """Instagram often rate-limits anonymous access. When that happens, wait and try again."""
    waits = [0] + list(a.rate_limit_waits)
    for attempt, wait in enumerate(waits):
        if wait and not sys.stdout.isatty():
            log(f"  rate-limited - waiting {wait}s before retry {attempt}/{len(waits) - 1}")
        for left in range(wait, 0, -1):
            bar.update("download", 0, f"rate-limited, retry {attempt}/{len(waits) - 1} in {left}s")
            time.sleep(1)
        try:
            return get_video(source, workdir, yt_dlp, a.cookies_from_browser,
                             lambda f: bar.update("download", f))
        except RuntimeError as e:
            if attempt == len(waits) - 1 or not RATE_LIMIT_RE.search(str(e)):
                raise


def process(source, a, tools):
    workdir = Path(tempfile.mkdtemp(prefix="v2t_"))
    result = {"status": "OK", "duration": 0, "lang": "-", "speech": [],
              "speech_note": "", "ocr": [], "ocr_note": "", "interval": None}

    use_ocr = bool(tools["tesseract"]) and not a.no_ocr
    bar = Bar({"download": 10, "audio": 2, "speech": 43, "frames": 5, "ocr": 40})
    if os.path.exists(source):
        bar.drop("download")  # local file: nothing to download
    if not use_ocr:
        bar.drop("frames")
        bar.drop("ocr")

    completed = False
    try:
        bar.update("download", 0)
        video = get_video_with_retry(source, workdir, tools["yt_dlp"], a, bar)
        result["duration"] = probe_duration(video)
        duration = result["duration"]

        # ---- spoken words ----
        if not has_audio(video):
            result["speech_note"] = "[no audio track in this video]"
            bar.drop("audio")
            bar.drop("speech")
        else:
            wav = workdir / "audio.wav"
            bar.update("audio", 0)
            if not extract_audio(video, wav, duration, lambda f: bar.update("audio", f)):
                result["speech_note"] = "[could not extract audio]"
                bar.drop("speech")
            elif is_silent(wav):
                result["speech_note"] = "[no speech - audio is silent]"
                bar.drop("speech")
            else:
                bar.update("speech", 0)
                segs, lang = transcribe(wav, tools["whisper"], a.model, a.language,
                                        str(a.threads),
                                        lambda f: bar.update("speech", f))
                result["lang"] = lang
                result["speech"] = segs
                if not segs:
                    result["speech_note"] = "[no speech detected - music or sound only]"

        # ---- on-screen text ----
        if use_ocr:
            interval = pick_interval(duration, a.ocr_interval)
            result["interval"] = interval
            bar.update("frames", 0)
            result["ocr"] = ocr_video(
                video, workdir, tools["tesseract"], tools["ocr_langs"], interval,
                str(a.psm), a.workers, duration,
                lambda f: bar.update("frames", f),
                lambda done, total: bar.update("ocr", done / total, f"{done}/{total} frames"),
            )
            if not result["ocr"]:
                result["ocr_note"] = "[no on-screen text found]"
        else:
            result["ocr_note"] = "[OCR skipped]"
        completed = True
    except Exception as e:  # keep going with the next link
        result["status"] = f"FAILED - {e}"
    finally:
        bar.finish(completed and result["status"] == "OK")
        if a.keep_temp:
            log(f"  (kept temp files in {workdir})")
        else:
            shutil.rmtree(workdir, ignore_errors=True)
    return result


def format_block(index, source, r):
    bar = "=" * 72
    out = [bar, f"[{index}] {source}", f"STATUS: {r['status']}"]
    if r["status"] != "OK":
        out += [bar, ""]
        return "\n".join(out)

    out.append(f"DURATION: {fmt_time(r['duration'])}   SPOKEN LANGUAGE: {r['lang']}")
    out += [bar, "", "--- SPOKEN TRANSCRIPT ---"]
    if r["speech"]:
        for s, e, text in r["speech"]:
            out.append(f"[{fmt_time(s)} - {fmt_time(e)}] {text}")
    else:
        out.append(r["speech_note"] or "[no speech]")

    out += ["", "--- ON-SCREEN TEXT (OCR, approximate) ---"]
    if r["ocr"]:
        for e in r["ocr"]:
            span = (f"[{fmt_time(e['start'])}]" if e["start"] == e["last"]
                    else f"[{fmt_time(e['start'])} - {fmt_time(e['last'])}]")
            out.append(f"{span} {e['text']}")
    else:
        out.append(r["ocr_note"] or "[no on-screen text found]")
    out += ["", ""]
    return "\n".join(out)


# ------------------------------------------------"interval"] = interval
            log(f"  reading on-screen text (1 frame every {interval}s) ...")
            result["ocr"] = ocr_video(video, workdir, tools["tesseract"],
                                      tools["ocr_langs"], interval, str(a.psm), a.workers)
            if not result["ocr"]:
                result["ocr_note"] = "[no on-screen text found]"
        else:
            result["ocr_note"] = "[OCR skipped]"
    except Exception as e:  # keep going with the next link
        result["status"] = f"FAILED - {e}"
    finally:
        if a.keep_temp:
            log(f"  (kept temp files in {workdir})")
        else:
            shutil.rmtree(workdir, ignore_errors=True)
    return result


def format_block(index, source, r):
    bar = "=" * 72
    out = [bar, f"[{index}] {source}", f"STATUS: {r['status']}"]
    if r["status"] != "OK":
        out += [bar, ""]
        return "\n".join(out)

    out.append(f"DURATION: {fmt_time(r['duration'])}   SPOKEN LANGUAGE: {r['lang']}")
    out += [bar, "", "--- SPOKEN TRANSCRIPT ---"]
    if r["speech"]:
        for s, e, text in r["speech"]:
            out.append(f"[{fmt_time(s)} - {fmt_time(e)}] {text}")
    else:
        out.append(r["speech_note"] or "[no speech]")

    out += ["", "--- ON-SCREEN TEXT (OCR, approximate) ---"]
    if r["ocr"]:
        for e in r["ocr"]:
            span = (f"[{fmt_time(e['start'])}]" if e["start"] == e["last"]
                    else f"[{fmt_time(e['start'])} - {fmt_time(e['last'])}]")
            out.append(f"{span} {e['text']}")
    else:
        out.append(r["ocr_note"] or "[no on-screen text found]")
    out += ["", ""]
    return "\n".join(out)


# ----------------------------------------------------------------------------
# setup checks + main
# ----------------------------------------------------------------------------
def pick_model(requested):
    if requested:
        return requested
    cache = Path.home() / ".cache" / "whisper"
    for name in ("ggml-small.bin", "ggml-base.bin", "ggml-medium.bin"):
        if (cache / name).exists():
            return str(cache / name)
    return None


def tesseract_langs(tesseract, wanted):
    r = run([tesseract, "--list-langs"])
    have = {l.strip() for l in (r.stdout or "").splitlines()[1:] if l.strip()}
    use = [l for l in wanted if l in have]
    missing = [l for l in wanted if l not in have]
    return "+".join(use), missing


def read_links(path):
    links = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                links.append(line)
    return links


def main():
    ap = argparse.ArgumentParser(description="Links in a txt -> transcript + on-screen text in a txt. No LLM.")
    ap.add_argument("input", nargs="?", default="links.txt", help="txt file with one link per line")
    ap.add_argument("-o", "--output", default="transcripts.txt", help="output txt file")
    ap.add_argument("--model", help="path to a multilingual whisper model (ggml-*.bin)")
    ap.add_argument("--language", default="auto", help="whisper language: auto, hi, en ... (default auto)")
    ap.add_argument("--ocr-langs", default="eng,hin", help="tesseract languages, comma separated (default eng,hin)")
    ap.add_argument("--ocr-interval", type=int, default=None,
                    help="seconds between OCR frames (default: 1s up to 3 min, 5s up to 20 min, else 10s)")
    ap.add_argument("--psm", type=int, default=11, help="tesseract page segmentation mode (default 11 = sparse text)")
    ap.add_argument("--no-ocr", action="store_true", help="skip on-screen text, transcript only")
    ap.add_argument("--cookies-from-browser", metavar="BROWSER",
                    help="e.g. chrome - use your browser login for posts that need it")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                    help="whisper threads")
    ap.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 4)),
                    help="parallel OCR workers")
    ap.add_argument("--keep-temp", action="store_true", help="do not delete downloaded files")
    a = ap.parse_args()

    # ---- check tools ----
    tools = {
        "yt_dlp": find_tool(["yt-dlp"]),
        "whisper": find_tool(["whisper-cli", "whisper-cpp"],
                             [r"C:\whisper\whisper-cli.exe"]),
        "tesseract": find_tool(["tesseract"],
                               [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]),
    }
    problems = []
    if not tools["yt_dlp"]:
        problems.append("yt-dlp not found")
    if not find_tool(["ffmpeg"]) or not find_tool(["ffprobe"]):
        problems.append("ffmpeg / ffprobe not found")
    if not tools["whisper"]:
        problems.append("whisper-cli not found")
    a.model = pick_model(a.model)
    if not a.model or not Path(a.model).exists():
        problems.append("whisper model not found (put ggml-small.bin or ggml-base.bin in ~/.cache/whisper/)")
    elif ".en." in Path(a.model).name and a.language != "en":
        problems.append(f"{Path(a.model).name} is English-only; use the multilingual model (no '.en') for Hindi")
    if problems:
        sys.exit("Setup problem:\n  - " + "\n  - ".join(problems))

    tools["ocr_langs"] = ""
    if a.no_ocr:
        pass
    elif not tools["tesseract"]:
        log("WARNING: tesseract not found - on-screen text will be skipped.")
    else:
        langs, missing = tesseract_langs(tools["tesseract"], a.ocr_langs.split(","))
        if missing:
            log(f"WARNING: tesseract language pack missing: {', '.join(missing)} "
                "(install it, e.g. tesseract-ocr-hin / tesseract-lang)")
        if langs:
            tools["ocr_langs"] = langs
        else:
            log("WARNING: no usable tesseract language - on-screen text will be skipped.")
            tools["tesseract"] = None

    if not Path(a.input).exists():
        sys.exit(f"Input file not found: {a.input}")
    links = read_links(a.input)
    if not links:
        sys.exit(f"No links found in {a.input}")

    log(f"{len(links)} link(s) to process. Model: {Path(a.model).name}")
    with open(a.output, "w", encoding="utf-8-sig") as out:
        out.write(f"Transcripts generated {datetime.now():%Y-%m-%d %H:%M}\n")
        out.write(f"Source list: {a.input}   Videos: {len(links)}\n\n")
        for i, link in enumerate(links, 1):
            log(f"[{i}/{len(links)}] {link}")
            result = process(link, a, tools)
            out.write(format_block(i, link, result) + "\n")
            out.flush()  # progress is saved after every video
            log(f"  -> {result['status']}")
    log(f"\nDone. Output: {a.output}")


if __name__ == "__main__":
    main()
