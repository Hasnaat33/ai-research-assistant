# AI Research Assistant

Give it a research question and a set of sources. It **gathers** the sources,
**synthesizes** the key findings into cited bullet points, then runs a
**verification pass** that cross-checks every claim against the source it
cites — flagging anything it can't ground.

Built to explore reliable AI-assisted research: prompt design, source
synthesis, and model-output evaluation.

## The problem

Researching any topic across dozens of sources takes hours, and raw
AI summaries often hallucinate facts or drop citations. Speed without
trust isn't useful.

## The approach

1. **Gather** — fetch URLs (or read local text files) and extract clean text.
2. **Synthesize** — rank sentences by relevance to the research question and
   compose cited findings. An optional `--llm` mode uses any
   OpenAI-compatible API for abstractive synthesis instead.
3. **Verify** — every finding is cross-checked: its distinctive terms must
   appear in the cited source, otherwise it's flagged ⚠ UNVERIFIED.

## The outcome

A markdown report where each finding carries its source citation and a
grounding verdict — research drafts in minutes that stay traceable to real
sources.

## Usage

```bash
pip install -r requirements.txt

# offline demo (no network, no API key needed)
python research.py --demo

# your own question + sources
python research.py "What is prompt engineering?" \
    --urls https://example.com/article1 https://example.com/article2

# local files as sources
python research.py "Summarize the Q3 notes" --files notes.txt report.txt

# LLM-powered synthesis (needs OPENAI_API_KEY; works with any
# OpenAI-compatible endpoint via OPENAI_BASE_URL)
python research.py --demo --llm
```

## Project structure

- `research.py` — the whole pipeline: gather → synthesize → verify → report
- `requirements.txt` — dependencies (only needed for URL fetching niceties)

## Skills demonstrated

Python · Prompt Engineering · Large Language Models · Generative AI ·
Source Evaluation · Technical Writing
