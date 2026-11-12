from common.checks import head_keys, load_results

from typing import TypedDict

import matplotlib.pyplot as plt

NUM_LAYERS = 12

class Results(TypedDict):
    clean_loss: float
    zero_ablation: dict[str, float]
    mean_ablation: dict[str, float]

def results():
    return load_results(__file__, ["clean_loss", "zero_ablation", "mean_ablation"])

def split_key(key: str)-> tuple[int, int]:
    layer, head = map(int, key[1:].split("H"))
    return (layer, head)

def main():
    r:Results = results()
    _, zero_ablation, mean_ablation = r['clean_loss'], r['zero_ablation'], r['mean_ablation']

    total_layer_loss = {"zero_ablation": [0.0 for _ in range(NUM_LAYERS)], 
                       "mean_ablation": [0.0 for _ in range(NUM_LAYERS)]}

    for key in zero_ablation:
        layer, _ = split_key(key)
        total_layer_loss["zero_ablation"][layer] += zero_ablation[key]

    for key in mean_ablation:
        layer, _ = split_key(key)
        total_layer_loss["mean_ablation"][layer] += mean_ablation[key]

    layers = range(NUM_LAYERS)
    width = 0.4
    plt.bar([l - width / 2 for l in layers], total_layer_loss["zero_ablation"], width, label="zero")
    plt.bar([l + width / 2 for l in layers], total_layer_loss["mean_ablation"], width, label="mean")
    plt.xticks(list(layers))
    plt.xlabel("layer")
    plt.ylabel("total loss increase")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

