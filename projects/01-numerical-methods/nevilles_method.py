"""Standalone command-line runner for Neville's method, run from inside this project.

Run with no arguments to reproduce the assignment's own data and its expected result,
``f(1.5) = 0.5118276664``:

    python nevilles_method.py

Or supply custom data (PowerShell line-continuation shown; drop the backticks on bash/zsh):

    python nevilles_method.py `
      --x "1.0,1.3,1.6,1.9,2.2,2.5" `
      --y "0.7651977,0.6200860,0.4554022,0.2818186,0.1103623,-0.0483838" `
      --target 1.5 `
      --csv neville_complete_table.csv

The recurrence and the CLI itself both live in ``quant_numerical/neville.py`` (this
project's tested package, also used by the Gradio interface in ``app.py`` and by
``tests/test_neville.py``) -- that one file needs only numpy beyond the standard library,
so it can be downloaded and run completely on its own, on any machine, independent of this
project's folder layout. This script is a convenience wrapper for running it from inside
this project directory instead, next to ``examples.py``, imported the same way that file
already imports the package so it works whether or not the package has been
``pip install``-ed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from quant_numerical.neville import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
