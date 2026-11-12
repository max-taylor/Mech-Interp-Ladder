from typing import TypedDict
from common.checks import load_results
import matplotlib.pyplot as plt
import numpy as np

NUM_LAYERS = 12
NUM_HEADS = 12

class Results(TypedDict):
    clean_loss: float
    zero_ablation: dict[str, float]
    mean_ablation: dict[str, float]

def results():
    return load_results(__file__, ["clean_loss", "zero_ablation", "mean_ablation"])

def main():
    r:Results = results()
    _, zero_ablation, mean_ablation = r['clean_loss'], r['zero_ablation'], r['mean_ablation']

    grid = np.zeros((NUM_LAYERS, NUM_HEADS))

    for layer in range(NUM_LAYERS):
        for head in range(NUM_HEADS):
            head_key = f"L{layer}H{head}"
            zero = zero_ablation[head_key]
            mean = mean_ablation[head_key]
            delta = zero - mean
            grid[layer][head] = delta

    limit = np.abs(grid).max()
    plt.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit, origin="lower")
    plt.colorbar(label="zero − mean loss increase")
    plt.xlabel("head")
    plt.ylabel("layer")
    plt.show()


if __name__ == "__main__":
    main()

