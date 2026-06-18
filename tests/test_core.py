"""Core tests — GPU-free and network-free.

The only thing mocked is the Groq HTTP call: a `ScriptedClient` returns a
pre-baked sequence of model responses. Everything else is real — the sandbox
executes real pandas/matplotlib against real DataFrames, and the agent loop
assembles a real trace. (Live Groq is exercised on the HF Space, where the call
originates from a region Groq allows.)
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from autoanalyst.agent import Analyst
from autoanalyst.dataio import load_table, schema_summary
from autoanalyst.sandbox import PythonSandbox


# --- a stand-in for the Groq client -------------------------------------------
def _tool_call(call_id: str, code: str):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name="run_python",
                                 arguments=json.dumps({"code": code})),
    )


def _response(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class ScriptedClient:
    """Returns a fixed list of responses in order, recording the calls made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


# --- sandbox -------------------------------------------------------------------
def test_sandbox_captures_stdout_and_chart():
    sb = PythonSandbox(pd.DataFrame({"a": [1, 2, 3]}))
    r = sb.run("print(df['a'].sum())\nplt.plot([1, 2, 3])")
    assert r.error is None
    assert r.stdout.strip() == "6"
    assert len(r.charts) == 1 and len(r.charts[0]) > 100  # a real base64 PNG


def test_sandbox_blocks_dangerous_import():
    sb = PythonSandbox(pd.DataFrame({"a": [1]}))
    r = sb.run("import os; os.system('echo nope')")
    assert r.error and "not allowed" in r.error


def test_sandbox_reports_runtime_error():
    sb = PythonSandbox(pd.DataFrame({"a": [1]}))
    r = sb.run("print(df['missing'])")
    assert r.error and "KeyError" in r.error


def test_sandbox_persists_namespace():
    sb = PythonSandbox(pd.DataFrame({"a": [1, 2]}))
    sb.run("total = int(df['a'].sum())")
    r = sb.run("print(total * 10)")
    assert r.stdout.strip() == "30"


# --- dataio --------------------------------------------------------------------
def test_schema_summary_mentions_shape_and_columns():
    s = schema_summary(pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}), "mini")
    assert "2 rows x 2 columns" in s
    assert "x :" in s and "y :" in s


def test_load_table_reads_sample_if_present():
    p = Path(__file__).resolve().parent.parent / "samples" / "titanic.csv"
    if not p.exists():
        pytest.skip("samples not built")
    df, schema = load_table(p)
    assert df.shape[0] > 800 and "Survived" in df.columns
    assert "Survived" in schema


# --- agent loop (real sandbox, scripted model) ---------------------------------
def test_agent_runs_real_code_then_answers():
    df = pd.DataFrame({"pclass": [1, 1, 2, 3, 3, 3],
                       "survived": [1, 1, 0, 0, 1, 0]})
    code = (
        "rate = df.groupby('pclass')['survived'].mean()\n"
        "print(rate)\n"
        "rate.plot(kind='bar'); plt.title('survival by class')"
    )
    client = ScriptedClient([
        _response(tool_calls=[_tool_call("c1", code)]),
        _response(content="First class survived most often; third class least."),
    ])
    analyst = Analyst(df, schema_summary(df, "t"), client=client)
    result = analyst.run("survival by class?")

    assert result.stopped == "answered"
    assert result.steps_used == 1
    assert "pclass" in result.trace[0].stdout
    assert result.trace[0].charts  # the bar chart was captured
    assert "third" in result.answer.lower()
    # the model was given the run_python tool on the first turn
    assert client.calls[0].get("tools")


def test_agent_hits_step_limit_and_forces_answer():
    df = pd.DataFrame({"a": [1, 2, 3]})
    looping = _response(tool_calls=[_tool_call("c", "print(df['a'].mean())")])
    client = ScriptedClient(
        [looping, looping, _response(content="The mean is 2.")]
    )
    analyst = Analyst(df, schema_summary(df), client=client, max_steps=2)
    result = analyst.run("never stops")
    assert result.stopped == "step_limit"
    assert result.steps_used == 2
    assert "mean is 2" in result.answer
    # the final, tool-less call must not offer tools
    assert "tools" not in client.calls[-1]
