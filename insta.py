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
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

TS_RE = re.compile(
    r"\[(\d+):(\d+):(\d+)[.,](\d+)\s*-->\s*(\d+):(\d+):(\d+)[.,](\d+)\]\s*(.*)"
)
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


def fmt_time(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# step 1: get the video
# ----------------------------------------------------------------------------
def get_video(source, workdir, yt_dlp, cookies_browser):
    if os.path.exists(source):
        return Path(source)

    out = workdir / "video.%(ext)s"
    cmd = [
        yt_dlp, "--no-playlist", "--no-warnings",
        "-f", "bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format", "mp4",
        "-o", str(out),
    ]
    if cookies_browser:
        cmd += ["--cookies-from-browser", cookies_browser]
    cmd.append(source)

    r = run(cmd)
    files = [p for p in workdir.glob("video.*") if p.suffix not in (".part", ".ytdl")]
    if r.returncode != 0 or not files:
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        raise RuntimeError(
            "download failed: " + " | ".join(tail)
            + "  (if it needs a login, re-run with --cookies-from-browser chrome)"
        )
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


# ----------------------------------------------------------------------------
# step 2: spoken words (whisper)
# ----------------------------------------------------------------------------
def extract_audio(video, wav):
    r = run(["ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", wav])
    return r.returncode == 0 and Path(wav).exists()


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


def transcribe(wav, whisper, model, lang, threads):
    r = run([whisper, "-m", model, "-f", wav, "-l", lang, "-t", threads])
    if r.returncode != 0:
        raise RuntimeError("whisper failed: " + (r.stderr or "").strip()[-300:])

    detected = re.search(r"auto-detected language:\s*(\w+)", r.stderr or "")
    segments = []
    for line in r.stdout.splitlines():
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


def extract_frames(video, frames_dir, interval):
    frames_dir.mkdir(exist_ok=True)
    r = run(["ffmpeg", "-y", "-i", video, "-vf", f"fps=1/{interval}",
             "-q:v", "2", str(frames_dir / "f_%05d.jpg")])
    if r.returncode != 0:
        raise RuntimeError("frame extraction failed")
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


def ocr_video(video, workdir, tesseract, langs, interval, psm, workers):
    frames = extract_frames(video, workdir / "frames", interval)
    if not frames:
        return []
    jobs = [(tesseract, f, langs, psm) for f in frames]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(ocr_frame, jobs))
    per_frame = [(i * interval, lines) for i, lines in enumerate(results)]
    return merge_ocr(per_frame, interval)


# ----------------------------------------------------------------------------
# per-link pipeline
# ----------------------------------------------------------------------------
def process(source, a, tools):
    workdir = Path(tempfile.mkdtemp(prefix="v2t_"))
    result = {"status": "OK", "duration": 0, "lang": "-", "speech": [],
              "speech_note": "", "ocr": [], "ocr_note": "", "interval": None}
    try:
        log("  downloading ...")
        video = get_video(source, workdir, tools["yt_dlp"], a.cookies_from_browser)
        result["duration"] = probe_duration(video)

        # ---- spoken words ----
        if not has_audio(video):
            result["speech_note"] = "[no audio track in this video]"
        else:
            wav = workdir / "audio.wav"
            if not extract_audio(video, wav):
                result["speech_note"] = "[could not extract audio]"
            elif is_silent(wav):
                result["speech_note"] = "[no speech - audio is silent]"
            else:
                log("  transcribing ...")
                segs, lang = transcribe(wav, tools["whisper"], a.model, a.language,
                                        str(a.threads))
                result["lang"] = lang
                result["speech"] = segs
                if not segs:
                    result["speech_note"] = "[no speech detected - music or sound only]"

        # ---- on-screen text ----
        if tools["tesseract"] and not a.no_ocr:
            interval = pick_interval(result["duration"], a.ocr_interval)
            result["interval"] = interval
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
