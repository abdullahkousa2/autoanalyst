"""The agent loop.

`Analyst.run(question)` drives a Groq chat model through a tool-calling loop:
the model writes Python, we execute it in a `PythonSandbox`, feed the printed
output (and a note about any chart) back, and repeat until the model stops
calling the tool and answers in plain English — or we hit the step limit.

Every executed snippet becomes a `Step` in the returned trace, which is what the
UI renders so you can watch the agent think.
"""
from __future__ import annotations

import json
import os

from .prompts import RUN_PYTHON_TOOL, SYSTEM_PROMPT, initial_user_message
from .sandbox import PythonSandbox, SandboxResult

DEFAULT_MODEL = os.environ.get("AUTOANALYST_MODEL", "llama-3.3-70b-versatile")
DEFAULT_MAX_STEPS = 8


class Step:
    """One executed snippet in the trace."""

    __slots__ = ("n", "code", "stdout", "error", "charts")

    def __init__(self, n: int, code: str, result: SandboxResult):
        self.n = n
        self.code = code
        self.stdout = result.stdout
        self.error = result.error
        self.charts = result.charts

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "code": self.code,
            "stdout": self.stdout,
            "error": self.error,
            "charts": self.charts,
        }


class AnalysisResult:
    def __init__(self, answer: str, trace: list[Step], steps_used: int, stopped: str):
        self.answer = answer
        self.trace = trace
        self.steps_used = steps_used
        self.stopped = stopped  # "answered" | "step_limit" | "error"

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "trace": [s.to_dict() for s in self.trace],
            "steps_used": self.steps_used,
            "stopped": self.stopped,
        }


def _make_client():
    """Build a Groq client lazily so importing this module needs no API key."""
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Put it in a .env file or your environment "
            "(create a free key at https://console.groq.com)."
        )
    return Groq(api_key=key)


def _assistant_to_dict(msg) -> dict:
    """Serialize a model message (with tool calls) back into the messages list."""
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in (msg.tool_calls or [])
        ],
    }


def _extract_code(tool_call) -> str:
    """Pull the `code` argument out of a tool call, tolerating loose JSON."""
    raw = tool_call.function.arguments or "{}"
    try:
        return json.loads(raw).get("code", "")
    except (json.JSONDecodeError, AttributeError):
        # Open models occasionally hand back the code unwrapped — run it anyway;
        # if it's broken the model will see the error and retry.
        return raw


def _tool_result_text(res: SandboxResult) -> str:
    parts = []
    parts.append(
        "Output:\n" + res.stdout.strip()
        if res.stdout.strip()
        else "(no output — remember to print() what you want to see)"
    )
    if res.error:
        parts.append("Error:\n" + res.error.strip())
    if res.charts:
        parts.append(f"[{len(res.charts)} chart(s) rendered for the user]")
    return "\n\n".join(parts)


class Analyst:
    def __init__(
        self,
        df,
        schema_summary: str,
        client=None,
        model: str = DEFAULT_MODEL,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        self.sandbox = PythonSandbox(df)
        self.schema = schema_summary
        self.client = client or _make_client()
        self.model = model
        self.max_steps = max_steps

    def _chat(self, messages: list, use_tools: bool = True, max_tokens: int = 1500):
        kwargs = dict(model=self.model, messages=messages,
                      temperature=0.0, max_tokens=max_tokens)
        if use_tools:
            kwargs.update(tools=[RUN_PYTHON_TOOL], tool_choice="auto")
        return self.client.chat.completions.create(**kwargs)

    def run_iter(self, question: str):
        """Drive the loop as a generator.

        Yields ("step", Step) after each executed snippet and finally
        ("final", AnalysisResult). This is what the SSE endpoint streams so the
        UI can render the agent thinking live.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_user_message(self.schema, question)},
        ]
        trace: list[Step] = []

        for _ in range(self.max_steps):
            resp = self._chat(messages)
            msg = resp.choices[0].message

            if not msg.tool_calls:
                answer = (msg.content or "").strip()
                yield "final", AnalysisResult(answer, trace, len(trace), "answered")
                return

            messages.append(_assistant_to_dict(msg))
            for tc in msg.tool_calls:
                code = _extract_code(tc)
                result = self.sandbox.run(code)
                step = Step(len(trace) + 1, code, result)
                trace.append(step)
                yield "step", step
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "run_python",
                    "content": _tool_result_text(result),
                })

        # Out of steps — force a final plain-English answer with no more tools.
        messages.append({
            "role": "user",
            "content": "You've reached the step limit. Give your best final "
            "answer now in plain English, using what you've already found.",
        })
        resp = self._chat(messages, use_tools=False, max_tokens=800)
        answer = (resp.choices[0].message.content or "").strip()
        yield "final", AnalysisResult(answer, trace, len(trace), "step_limit")

    def run(self, question: str, on_step=None) -> AnalysisResult:
        """Run the loop to completion. `on_step(step)` fires per executed snippet."""
        final: AnalysisResult | None = None
        for kind, payload in self.run_iter(question):
            if kind == "step" and on_step:
                on_step(payload)
            elif kind == "final":
                final = payload
        return final
