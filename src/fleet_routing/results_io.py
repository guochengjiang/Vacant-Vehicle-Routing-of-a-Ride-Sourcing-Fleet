"""Small result helpers; load only your own trusted pickle files."""
import pickle
from pathlib import Path


def save_result(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved: {path}')
    return path


def load_result(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)
