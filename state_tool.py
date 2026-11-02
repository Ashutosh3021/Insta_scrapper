#!/usr/bin/env python3
"""
state_tool.py - keeps track of which reels are done, so a run can stop and a later run can continue.

State file  state_<id>.txt  (CSV):

    Reel_no,link,status,tries
    1,https://www.instagram.com/reel/AAA/,1,0

  status 1 = done
  status 0 = gave up (failed twice for a real reason, e.g. deleted reel) - never tried again
  status 2 = not finished (never ran, cut off by the time limit, rate-limited, or failed once)
  tries    = how many times the reel really failed. At MAX_TRIES (2) the status becomes 0.
             Instagram rate limits and runs that were cut off do NOT count as a try.

Commands (the workflow calls these):
  init   make a new state file from a reels file
  plan   split the status-2 reels into chunk input files
  apply  read the chunk results, save finished transcripts, update the state file
  build  join every finished transcript (all runs so far) into one text file
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ["Reel_no", "link", "status", "tries"]
BAR = "=" * 72
BLOCK_SPLIT = re.compile(r"\n={72}\n(?=\[\d+\] )")


def eprint(*a):
    print(*a, file=sys.stderr)


def norm(link):
    return link.strip().rstrip("/")


# ----------------------------------------------------------------------------
# state file
# ----------------------------------------------------------------------------
def load_state(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in (reader.fieldnames or [])]
        for r in reader:
            if not (r.get("Reel_no") or "").strip():
                continue
            rows.append({
                "no": int(r["Reel_no"]),
                "link": r["link"].strip(),
                "status": int(r["status"]),
                "tries": int((r.get("tries") or "0").strip() or 0),  # a hand-made 3-column file still works
            })
    return rows


def save_state(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(FIELDS)
        for r in rows:
            w.writerow([r["no"], r["link"], r["status"], r["tries"]])


def read_reels(path):
    links, seen = [], set()
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key = norm(line)
            if key in seen:
                eprint(f"duplicate link ignored: {line}")
                continue
            seen.add(key)
            links.append(line)
    return links


# ----------------------------------------------------------------------------
# init
# ----------------------------------------------------------------------------
def cmd_init(a):
    links = read_reels(a.reels)
    if not links:
        sys.exit(f"No links found in {a.reels}")
    save_state(a.state, [{"no": i, "link": l, "status": 2, "tries": 0} for i, l in enumerate(links, 1)])
    eprint(f"created {a.state} with {len(links)} reels (all status 2 = not finished)")


# ----------------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------------
def cmd_plan(a):
    rows = load_state(a.state)
    todo = [r for r in rows if r["status"] == 2]
    total = len(rows)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not todo:
        # a placeholder keeps the workflow's matrix valid; the chunk jobs are skipped when todo=0
        print('matrix={"chunk":[0]}')
        print("todo=0")
        print(f"total={total}")
        eprint("nothing left to do")
        return

    k = max(1, min(int(a.chunks), len(todo), 100))
    base, extra = divmod(len(todo), k)
    i = 0
    for c in range(1, k + 1):
        size = base + (1 if c <= extra else 0)
        part = todo[i:i + size]
        i += size
        (out / f"chunk_{c}_input.txt").write_text(
            "".join(f"{r['no']},{r['link']}\n" for r in part), encoding="utf-8")
    print("matrix=" + json.dumps({"chunk": list(range(1, k + 1))}))
    print(f"todo={len(todo)}")
    print(f"total={total}")
    eprint(f"{len(todo)} reels to do, split into {k} chunk(s) of {base}-{base + (1 if extra else 0)}")


# ----------------------------------------------------------------------------
# apply
# ----------------------------------------------------------------------------
def cmd_apply(a):
    rows = load_state(a.state)
    by_no = {r["no"]: r for r in rows}
    tdir = Path(a.transcripts_dir)
    tdir.mkdir(parents=True, exist_ok=True)
    cdir = Path(a.chunks_dir)

    done = counted = rate = ignored = 0
    notes = []  # (reel no, text) for this run's problems

    for f in sorted(cdir.glob("chunk_*.txt")):
        text = f.read_text(encoding="utf-8-sig", errors="replace")
        for b in BLOCK_SPLIT.split("\n" + text):
            m = re.match(r"\[(\d+)\] (.+)\n", b)
            if not m:
                continue
            no, link = int(m.group(1)), m.group(2).strip()
            r = by_no.get(no)
            if r is None or norm(r["link"]) != norm(link):
                ignored += 1
                eprint(f"ignored a block for reel {no}: it does not match the state file")
                continue
            if r["status"] != 2:
                continue  # already settled in an earlier run

            if re.search(r"^STATUS: OK\b", b, re.M):
                if "--- ON-SCREEN TEXT" not in b:
                    continue  # a block cut off in the middle of writing: treat as not done
                (tdir / f"reel_{no:04d}.txt").write_text(
                    f"# model: {a.model}\n{BAR}\n{b.rstrip()}\n", encoding="utf-8")
                r["status"] = 1
                done += 1
            elif re.search(r"^STATUS: FAILED", b, re.M):
                kind = re.search(r"^KIND: (\w+)", b, re.M)
                kind = kind.group(1) if kind else "error"
                reason = re.search(r"^STATUS: FAILED - (.*)$", b, re.M)
                reason = (reason.group(1) if reason else "")[:160]
                if kind == "rate_limit":
                    rate += 1
                    notes.append((no, f"rate-limited by the site (not counted as a try): {reason}"))
                else:
                    r["tries"] += 1
                    counted += 1
                    if r["tries"] >= a.max_tries:
                        r["status"] = 0
                        notes.append((no, f"gave up after {r['tries']} tries: {reason}"))
                    else:
                        notes.append((no, f"failed (try {r['tries']} of {a.max_tries}): {reason}"))

    save_state(a.state, rows)

    n1 = sum(1 for r in rows if r["status"] == 1)
    n0 = sum(1 for r in rows if r["status"] == 0)
    n2 = sum(1 for r in rows if r["status"] == 2)
    progress = (done + counted) > 0

    lines = [
        f"### State `{Path(a.state).name}`: {n1} done, {n0} gave up, {n2} not finished (of {len(rows)} reels)",
        "",
        f"This run: {done} finished, {counted} failed (counted), {rate} rate-limited (not counted).",
    ]
    if notes:
        lines += ["", "Problems in this run:"] + [f"- reel {no}: {t}" for no, t in notes]
    dead = [r for r in rows if r["status"] == 0]
    if dead:
        lines += ["", "Gave up on these (status 0, never retried unless you set the status back to 2):"]
        lines += [f"- reel {r['no']}: {r['link']}" for r in dead]
    if n2:
        lines += ["", f"{n2} reel(s) still have status 2 and will be picked up by the next run."]
    Path(a.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"remaining={n2}")
    print(f"progress={'true' if progress else 'false'}")
    print(f"done={n1}")
    eprint(f"finished {done}, counted failures {counted}, rate-limited {rate}, ignored blocks {ignored}")


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------
def cmd_build(a):
    rows = load_state(a.state)
    tdir = Path(a.transcripts_dir)
    n1 = sum(1 for r in rows if r["status"] == 1)
    n0 = sum(1 for r in rows if r["status"] == 0)
    n2 = sum(1 for r in rows if r["status"] == 2)

    out = [
        f"Transcripts for {Path(a.state).name}   built {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
        f"Reels: {len(rows)}   done: {n1}   gave up: {n0}   not finished: {n2}",
        "",
    ]
    for r in rows:
        f = tdir / f"reel_{r['no']:04d}.txt"
        if r["status"] == 1 and f.exists():
            out.append(f.read_text(encoding="utf-8-sig").rstrip() + "\n")
        elif r["status"] == 1:
            out.append(f"{BAR}\n[{r['no']}] {r['link']}\nSTATUS: DONE, but its transcript file is missing\n{BAR}\n")
        elif r["status"] == 0:
            out.append(f"{BAR}\n[{r['no']}] {r['link']}\nSTATUS: GAVE UP after {r['tries']} tries (status 0)\n{BAR}\n")
        else:
            out.append(f"{BAR}\n[{r['no']}] {r['link']}\nSTATUS: NOT FINISHED YET (status 2, tries so far: {r['tries']})\n{BAR}\n")
    Path(a.out).write_text("\n".join(out), encoding="utf-8-sig")
    eprint(f"wrote {a.out}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create a state file from a reels file")
    p.add_argument("--reels", required=True)
    p.add_argument("--state", required=True)
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("plan", help="split the unfinished reels into chunk input files")
    p.add_argument("--state", required=True)
    p.add_argument("--chunks", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("apply", help="read chunk results and update the state")
    p.add_argument("--state", required=True)
    p.add_argument("--chunks-dir", required=True)
    p.add_argument("--transcripts-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--max-tries", type=int, default=2)
    p.add_argument("--summary", required=True)
    p.set_defaults(fn=cmd_apply)

    p = sub.add_parser("build", help="join all finished transcripts into one file")
    p.add_argument("--state", required=True)
    p.add_argument("--transcripts-dir", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_build)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
