import random
from itertools import islice

from datasets import load_dataset

PILE_DATASET = "NeelNanda/pile-10k"

FACTUAL_PROMPTS: list[tuple[str, str]] = [
    ("Tokyo is the capital of", " Japan"),
    ("Paris is the capital of", " France"),
    ("The Statue of Liberty is in New", " York"),
    ("Barack Obama was the president of the United", " States"),
    ("Michael Jordan played the sport of", " basketball"),
    ("Shakespeare wrote Romeo and", " Juliet"),
    ("The opposite of hot is", " cold"),
    ("Mount Everest is the tallest mountain in the", " world"),
]

IOI_NAMES = [" Mary", " John", " Tom", " James", " Dan", " Sid", " Martin", " Amy"]
IOI_TEMPLATE = "When{a} and{b} went to the store,{s} gave a drink to"


def get_ioi_pairs(n_pairs: int = 8, seed: int = 0) -> list[dict[str, str]]:
    """IOI prompt pairs sharing one template, so token positions line up.

    clean:   "When A and B went to the store, B gave a drink to"  -> answer A
    corrupt: "When A and B went to the store, A gave a drink to"  -> answer B
    Each dict has keys clean, corrupt, correct (A), incorrect (B).
    """
    rng = random.Random(seed)
    pairs = []
    for _ in range(n_pairs):
        a, b = rng.sample(IOI_NAMES, 2)
        pairs.append(
            {
                "clean": IOI_TEMPLATE.format(a=a, b=b, s=b),
                "corrupt": IOI_TEMPLATE.format(a=a, b=b, s=a),
                "correct": a,
                "incorrect": b,
            }
        )
    return pairs


def load_pile_texts(n_docs: int = 200, min_chars: int = 500) -> list[str]:
    """First n_docs documents from pile-10k with at least min_chars characters."""
    ds = load_dataset(PILE_DATASET, split="train")
    return list(islice((row["text"] for row in ds if len(row["text"]) >= min_chars), n_docs))


QUOTE_WORDS = (
    "the cat dog house tree river car city friend teacher music book idea plan water fire "
    "green small quickly happy old new big red slow bright quiet loud early late "
    "runs sees likes finds makes takes brings keeps opens wants"
).split()


def make_quote_texts(n_texts: int = 400, seed: int = 0) -> list[str]:
    """Synthetic texts alternating unquoted and quoted runs of words from one shared vocabulary.

    Run lengths and the number of quotes vary, so position alone doesn't reveal the label.
    Example: 'river the green runs "old music opens" city friend "bright tree" keeps'
    """
    rng = random.Random(seed)

    def words() -> str:
        return " ".join(rng.choice(QUOTE_WORDS) for _ in range(rng.randint(1, 8)))

    texts = []
    for _ in range(n_texts):
        n_quotes = rng.randint(1, 3)
        parts = [words()]
        for _ in range(n_quotes):
            parts += [f'"{words()}"', words()]
        texts.append(" ".join(parts))
    return texts
