import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from quant_numerical import lagrange_interpolate  # noqa: E402


def main() -> None:
    output = Path(__file__).parent / "artifacts"
    output.mkdir(exist_ok=True)
    nodes = np.array([0.0, 0.5, 1.0])
    observed = np.exp(nodes)
    grid = np.linspace(0.0, 1.0, 300)
    estimate = lagrange_interpolate(nodes, observed, grid)
    error = np.abs(np.exp(grid) - estimate)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(grid, np.exp(grid), label="true $e^x$")
    axes[0].plot(grid, estimate, "--", label="quadratic interpolation")
    axes[0].scatter(nodes, observed, color="black", label="nodes")
    axes[0].set(title="Lagrange interpolation", xlabel="x", ylabel="f(x)")
    axes[0].legend()
    axes[1].plot(grid, error, color="firebrick")
    axes[1].set(title="Absolute interpolation error", xlabel="x", ylabel="absolute error")
    fig.tight_layout()
    fig.savefig(output / "lagrange_demo.png", dpi=160)
    print(f"Maximum absolute error: {error.max():.6g}")


if __name__ == "__main__":
    main()

