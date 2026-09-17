"""Gradio interface for Neville's method — polynomial interpolation with full calculation evidence.

Run locally with:

    python -m pip install -e ".[app]"
    python app.py

The interface imports the same tested `quant_numerical.neville` engine used by
`tests/test_neville.py` and `nevilles_method.py` (the standalone CLI runner) — the math is
never reimplemented here, only displayed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from quant_numerical.neville import (  # noqa: E402
    ASSIGNMENT_TARGET,
    ASSIGNMENT_X,
    ASSIGNMENT_Y,
    neville_interpolate,
    table_grid,
    write_csv,
)

PROJECT_DIR = Path(__file__).parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
ENGINE_SOURCE = PROJECT_DIR / "src" / "quant_numerical" / "neville.py"

EXPLANATION_MD = """
# Appeal: Sample Workings to Augment an Earlier Submitted Assignment

On my last submission for this Neville's Method question, I lost marks because I only handed
in the final answer with no working shown — there was nothing to check my process against.
This page accompanies that submission with the full working, computed live rather than typed
by hand, using the exact recurrence we covered in class:

```
Q[i, 0] = f(x_i)

Q[i, j] = ((x_target - x[i-j]) * Q[i, j-1] - (x_target - x[i]) * Q[i-1, j-1])
          / (x[i] - x[i-j])
```

Defaults below are the assignment's own data (x = 1.0 to 2.5, target x = 1.5, expected result
f(1.5) = 0.5118276664). Every working can be checked two ways: directly on this page — the
table and every substituted calculation below — or independently, using the CSV and Python
downloads next to the result.
"""

DOWNLOADS_NOTE = """
### Download and check independently

- **CSV** — not just a transcript: alongside the finished value, each row also has the
  recurrence's five numeric inputs in their own columns. Open it in Excel, build the formula
  yourself in a spare column from those columns, and confirm it matches — a ready-to-test
  solution, not only something to read.
- **Python source (`neville.py`)** — the real code that produced every number on this page,
  not hand-typed workings. It needs only `numpy` and nothing else from this project, so it can
  be downloaded on its own and run/validated on any other machine: `pip install numpy` then
  `python neville.py` reproduces the table above from scratch.
"""


def _parse_floats(raw: str) -> list[float]:
    values = [chunk.strip() for chunk in raw.split(",")]
    return [float(chunk) for chunk in values if chunk != ""]


def calculate(x_text: str, y_text: str, target_text: str):
    try:
        x_values = _parse_floats(x_text)
        y_values = _parse_floats(y_text)
        target = float(target_text)
        result = neville_interpolate(x_values, y_values, target)
    except ValueError as exc:
        error_md = (
            f"### Input error\n\n**{exc}**\n\nCorrect the values above and press Calculate again."
        )
        return error_md, None, "", None
    except Exception as exc:  # malformed number text, e.g. "1.0,,abc"
        error_md = f"### Input error\n\nCould not read the values above as numbers: **{exc}**"
        return error_md, None, "", None

    estimate_md = (
        f"### Result\n\n"
        f"**f({target:g}) ≈ {result.estimate:.10f}**\n\n"
        f"({len(result.steps)} calculations total: {len(x_values)} initial values "
        f"Q[i,0] plus {len(result.steps) - len(x_values)} recursive entries.)"
    )

    grid = table_grid(result)
    table_df = pd.DataFrame(grid[1:], columns=grid[0])

    steps_lines = ["### Every calculation step\n"]
    steps_lines.append("**Initial values:**\n")
    for step in result.steps:
        if step.j == 0:
            steps_lines.append(f"- `{step.formula}`")
    steps_lines.append("\n**Recursive calculations:**\n")
    for step in result.steps:
        if step.j != 0:
            steps_lines.append(f"- `{step.formula}`")
    steps_md = "\n".join(steps_lines)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    csv_path = write_csv(result, ARTIFACTS_DIR / "neville_complete_table.csv")

    return estimate_md, table_df, steps_md, str(csv_path)


with gr.Blocks(title="Appeal: Neville's Method Workings") as demo:
    gr.Markdown(EXPLANATION_MD)

    with gr.Row():
        with gr.Column(scale=1):
            x_input = gr.Textbox(
                label="x-values (comma-separated)",
                value=", ".join(f"{v:g}" for v in ASSIGNMENT_X),
            )
            y_input = gr.Textbox(
                label="f(x)-values (comma-separated)",
                value=", ".join(f"{v:.10g}" for v in ASSIGNMENT_Y),
            )
            target_input = gr.Textbox(
                label="Evaluation point (target x)", value=f"{ASSIGNMENT_TARGET:g}"
            )
            calculate_btn = gr.Button("Calculate", variant="primary")

        with gr.Column(scale=1):
            result_out = gr.Markdown()
            gr.Markdown(DOWNLOADS_NOTE)
            with gr.Row():
                csv_out = gr.File(label="Download the complete table as CSV")
                gr.File(
                    value=str(ENGINE_SOURCE),
                    label="Python source (neville.py) — runs standalone anywhere",
                )

    table_out = gr.Dataframe(label="Complete recursive table", interactive=False)
    steps_out = gr.Markdown()

    inputs = [x_input, y_input, target_input]
    outputs = [result_out, table_out, steps_out, csv_out]
    calculate_btn.click(fn=calculate, inputs=inputs, outputs=outputs)
    demo.load(fn=calculate, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch()
