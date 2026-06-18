---
title: AutoAnalyst
emoji: 📊
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 8000
pinned: true
license: mit
short_description: An agent that writes & runs its own data analysis
---

# 📊 AutoAnalyst

Ask a question in plain English → AutoAnalyst **writes Python, runs it against a real
dataset, draws charts, and explains the answer** — streaming every step live so you can
watch it think.

It's an agentic tool-use loop, not a one-shot prompt: the model gets a single
`run_python` tool and a live pandas session, then explores the data, computes the
result, and self-corrects when its code errors.

## How to use

1. Pick one of the bundled datasets (Titanic, tips, penguins, e-commerce sales).
2. Click an example question or type your own.
3. Hit **Analyze** and watch the steps appear — the code it wrote, the output, the
   charts — followed by a plain-English answer.

> Powered by **Groq · llama-3.3-70b** tool-calling + a sandboxed pandas/matplotlib
> executor. Uploads are disabled on this public Space (curated datasets only); run it
> [locally](https://github.com/abdullahkousa2/autoanalyst) to analyze your own files.

## Links

- 💻 **Source & docs:** [github.com/abdullahkousa2/autoanalyst](https://github.com/abdullahkousa2/autoanalyst)
- 📦 **Library:** `pip install autoanalyst`

Built with Groq · pandas · matplotlib · FastAPI · Server-Sent Events. No GPU required.
