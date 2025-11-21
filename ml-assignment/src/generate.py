import sys
import os

# Add the parent directory to sys.path so "src" can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)



from src.ngram_model import TrigramModel
from pathlib import Path


def main():
    model = TrigramModel()

    data_path = Path(__file__).parent.parent / "data" / "example_corpus.txt"
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    model.fit(text)

    generated = model.generate(max_length=50)
    print("Generated Text:")
    print(generated)


if __name__ == "__main__":
    main()
