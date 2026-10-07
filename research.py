#!/usr/bin/env python3
"""
AI Research Assistant
---------------------
Give it a research question and a set of sources (URLs or local text files).
It gathers the sources, synthesizes the key findings into cited bullet points,
and then runs a verification pass that cross-checks every claim against the
source it cites — flagging anything it cannot ground.

Usage:
    python research.py "What is prompt engineering?" --urls https://example.com/a https://example.com/b
    python research.py "What is prompt engineering?" --files notes.txt article.txt
    python research.py --demo            # runs offline on bundled sample sources
    python research.py --demo --llm      # same, but uses an LLM for the synthesis step
                                         # (needs OPENAI_API_KEY env var)

No API key is required for the default extractive mode.
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from datetime import date

# ----------------------------------------------------------------------------
# 1. GATHER — fetch sources
# ----------------------------------------------------------------------------

def fetch_url(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "AI-Research-Assistant/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    # strip scripts/styles, then tags — stdlib only, no extra dependency
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_file(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


DEMO_SOURCES = [
    (
        "Demo Source 1: Prompt Engineering Guide",
        """Prompt engineering is the practice of designing effective inputs for large
        language models. Clear, specific prompts produce more accurate outputs than vague
        ones. Techniques include giving the model a role, providing examples of desired
        output, and asking it to reason step by step before answering. Studies of model
        behavior show that structured prompts reduce hallucinations because the model is
        constrained to follow an explicit reasoning path.""",
    ),
    (
        "Demo Source 2: Evaluating AI Outputs",
        """Evaluating AI-generated text requires checking factual claims against trusted
        sources. A common failure mode is hallucination, where the model produces
        plausible but false statements. Best practice is to require citations for factual
        claims and to verify each claim independently. Prompt chaining — breaking a hard
        task into smaller prompted steps — improves reliability because errors can be
        caught at each stage.""",
    ),
]


def gather(urls, files, demo):
    """Return a list of (title, text) sources."""
    sources = []
    if demo:
        return list(DEMO_SOURCES)
    for i, url in enumerate(urls or [], 1):
        try:
            sources.append((f"Source {i}: {url}", fetch_url(url)))
        except Exception as e:  # noqa: BLE001 — report and continue with the rest
            print(f"  ! could not fetch {url}: {e}", file=sys.stderr)
    for path in files or []:
        try:
            sources.append((f"Source {len(sources)+1}: {os.path.basename(path)}", read_file(path)))
        except OSError as e:
            print(f"  ! could not read {path}: {e}", file=sys.stderr)
    return sources


# ----------------------------------------------------------------------------
# 2. SYNTHESIZE — turn sources into cited findings
# ----------------------------------------------------------------------------

STOPWORDS = set(
    "the a an and or of to in on for with is are was were be been it its this that "
    "these those as at by from we you he she they them his her their our your i".split()
)


def keywords(question):
    words = re.findall(r"[a-z]{3,}", question.lower())
    return [w for w in words if w not in STOPWORDS]


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 40]


def score_sentence(sentence, kw):
    words = set(re.findall(r"[a-z]{3,}", sentence.lower()))
    return sum(1 for k in kw if k in words)


def synthesize_extractive(question, sources, per_source=3):
    """Pick the most question-relevant sentences from each source."""
    kw = keywords(question)
    findings = []
    for idx, (title, text) in enumerate(sources, 1):
        ranked = sorted(
            ((score_sentence(s, kw), s) for s in split_sentences(text)),
            key=lambda x: x[0],
            reverse=True,
        )
        for sc, sent in ranked[:per_source]:
            if sc > 0:
                findings.append({"text": sent, "source": idx, "title": title})
    return findings


def synthesize_llm(question, sources, per_source=3):
    """Optional abstractive synthesis via any OpenAI-compatible API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    if not api_key:
        print("  ! OPENAI_API_KEY not set — falling back to extractive mode.", file=sys.stderr)
        return synthesize_extractive(question, sources, per_source)

    context = "\n\n".join(
        f"[Source {i}: {title}]\n{text[:4000]}" for i, (title, text) in enumerate(sources, 1)
    )
    prompt = (
        f"Answer the research question using ONLY the sources below. "
        f"Write {per_source * len(sources)} short bullet findings, each ending with "
        f"its citation like [Source 2]. If a point is not supported, say so.\n\n"
        f"Question: {question}\n\n{context}"
    )
    body = json.dumps({
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    reply = data["choices"][0]["message"]["content"]
    findings = []
    for line in reply.splitlines():
        line = line.strip().lstrip("-•* ").strip()
        m = re.search(r"\[Source (\d+)\]", line)
        if line and m:
            findings.append({"text": re.sub(r"\s*\[Source \d+\]", "", line), "source": int(m.group(1)), "title": ""})
    return findings or synthesize_extractive(question, sources, per_source)


# ----------------------------------------------------------------------------
# 3. VERIFY — cross-check every claim against its cited source
# ----------------------------------------------------------------------------

def verify(findings, sources):
    """Flag any finding whose distinctive words don't appear in its cited source."""
    checked = []
    for f in findings:
        src_idx = f["source"] - 1
        src_text = sources[src_idx][1].lower() if 0 <= src_idx < len(sources) else ""
        words = [w for w in re.findall(r"[a-z]{4,}", f["text"].lower()) if w not in STOPWORDS]
        distinctive = sorted(set(words), key=words.count, reverse=True)[:6]
        hits = sum(1 for w in distinctive if w in src_text)
        f["verified"] = hits >= max(2, len(distinctive) // 2)
        f["evidence_words"] = [w for w in distinctive if w in src_text][:4]
        checked.append(f)
    return checked


# ----------------------------------------------------------------------------
# 4. REPORT
# ----------------------------------------------------------------------------

def render_markdown(question, findings, sources):
    lines = [f"# Research: {question}", "", f"_Generated {date.today().isoformat()}_", ""]
    lines.append("## Key findings")
    for f in findings:
        mark = "✓" if f["verified"] else "⚠ UNVERIFIED"
        lines.append(f"- {f['text']} [Source {f['source']}] — {mark}")
    lines += ["", "## Sources"]
    for i, (title, _text) in enumerate(sources, 1):
        lines.append(f"{i}. {title}")
    unverified = [f for f in findings if not f["verified"]]
    if unverified:
        lines += ["", f"> {len(unverified)} finding(s) could not be grounded in the cited source — treat with caution."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="AI Research Assistant — gather, synthesize, verify.")
    ap.add_argument("question", nargs="?", default="", help="Research question")
    ap.add_argument("--urls", nargs="*", default=[], help="Source URLs to fetch")
    ap.add_argument("--files", nargs="*", default=[], help="Local text files to use as sources")
    ap.add_argument("--demo", action="store_true", help="Run offline on bundled sample sources")
    ap.add_argument("--llm", action="store_true", help="Use an LLM for synthesis (needs OPENAI_API_KEY)")
    ap.add_argument("--out", default="report.md", help="Output markdown file")
    args = ap.parse_args()

    if not args.demo and not args.urls and not args.files:
        ap.error("provide --urls / --files, or use --demo")
    question = args.question or "What is prompt engineering?"

    print(f"Q: {question}\n")
    print("1/3 gathering sources…")
    sources = gather(args.urls, args.files, args.demo)
    if not sources:
        sys.exit("no usable sources — aborting.")
    print(f"    {len(sources)} source(s) loaded.")

    print("2/3 synthesizing findings…")
    findings = (synthesize_llm if args.llm else synthesize_extractive)(question, sources)
    print(f"    {len(findings)} finding(s).")

    print("3/3 verifying claims against sources…")
    findings = verify(findings, sources)
    ok = sum(1 for f in findings if f["verified"])
    print(f"    {ok}/{len(findings)} grounded in cited sources.")

    report = render_markdown(question, findings, sources)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nDone — report written to {args.out}")
    print("-" * 60)
    print(report)


if __name__ == "__main__":
    main()
