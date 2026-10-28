"""Pré-processamento da base MR, fiel ao `process_data.py` original de Yoon Kim.

Gera `data/mr.pkl` com:
  - revs:   lista de frases {y, text, num_words, split}  (split = fold 0..9)
  - W:      matriz de embeddings word2vec (palavras fora do word2vec ~ U[-0.25, 0.25])
  - W2:     matriz totalmente aleatória U[-0.25, 0.25] (usada pelo CNN-rand)
  - word_idx_map: palavra -> índice (o índice 0 é reservado ao padding, vetor zero)
  - vocab:  palavra -> frequência de documento

Uso:
  python src/preprocess.py                       # baixa word2vec via gensim (~1,6 GB)
  python src/preprocess.py --w2v GoogleNews-vectors-negative300.bin.gz
  python src/preprocess.py --no-w2v              # só W2 (permite rodar apenas o CNN-rand)
"""
import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
K = 300  # dimensão dos vetores word2vec
SEED = 3435  # mesma semente usada no código original


def clean_str(string, trec=False):
    """Tokenização idêntica à do código original (inclusive os escapes `\\(`)."""
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    # No Python 2 original, " \( " na substituição vira o token literal "\(".
    string = re.sub(r"\(", r" \\( ", string)
    string = re.sub(r"\)", r" \\) ", string)
    string = re.sub(r"\?", r" \\? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if trec else string.strip().lower()


def build_data_cv(pos_file, neg_file, rng, cv=10):
    """Lê as frases e sorteia um fold (0..cv-1) para cada uma, como no original."""
    revs = []
    vocab = defaultdict(float)
    for path, label in ((pos_file, 1), (neg_file, 0)):
        # O original lia bytes no Python 2; latin-1 preserva 1 byte = 1 caractere.
        with open(path, encoding="latin-1") as f:
            for line in f:
                text = clean_str(line.strip())
                for word in set(text.split()):
                    vocab[word] += 1
                revs.append({
                    "y": label,
                    "text": text,
                    "num_words": len(text.split()),
                    "split": int(rng.randint(0, cv)),
                })
    return revs, vocab


def load_word2vec(vocab, w2v_path=None):
    """Retorna {palavra: vetor} só para as palavras do vocabulário da base."""
    from gensim.models import KeyedVectors

    if w2v_path:
        kv = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    else:
        import gensim.downloader as api

        kv = api.load("word2vec-google-news-300")
    return {w: np.asarray(kv[w], dtype="float32") for w in vocab if w in kv.key_to_index}


def add_unknown_words(word_vecs, vocab, rng, min_df=1, k=K):
    """0.25 faz os vetores aleatórios terem ~ a mesma variância dos pré-treinados."""
    for word in sorted(vocab):
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = rng.uniform(-0.25, 0.25, k).astype("float32")


def get_W(word_vecs, word_idx_map, k=K):
    W = np.zeros((len(word_idx_map) + 1, k), dtype="float32")  # linha 0 = padding
    for word, i in word_idx_map.items():
        W[i] = word_vecs[word]
    return W


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w2v", help="caminho do GoogleNews-vectors-negative300.bin(.gz)")
    parser.add_argument("--no-w2v", action="store_true", help="não carrega word2vec (só CNN-rand)")
    args = parser.parse_args()

    rng = np.random.RandomState(SEED)
    revs, vocab = build_data_cv(DATA_DIR / "rt-polarity.pos", DATA_DIR / "rt-polarity.neg", rng)
    word_idx_map = {w: i + 1 for i, w in enumerate(sorted(vocab))}
    max_l = max(r["num_words"] for r in revs)
    print(f"frases: {len(revs)} | vocabulário: {len(vocab)} | maior frase: {max_l} palavras")
    print(f"tamanho médio: {np.mean([r['num_words'] for r in revs]):.1f} palavras")

    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab, rng)
    W2 = get_W(rand_vecs, word_idx_map)

    W, n_pre = None, 0
    if not args.no_w2v:
        print("carregando word2vec (pode demorar alguns minutos)...")
        w2v = load_word2vec(vocab, args.w2v)
        n_pre = len(w2v)
        print(f"palavras presentes no word2vec: {n_pre}")
        add_unknown_words(w2v, vocab, rng)
        W = get_W(w2v, word_idx_map)

    out = DATA_DIR / "mr.pkl"
    with open(out, "wb") as f:
        pickle.dump({"revs": revs, "W": W, "W2": W2, "word_idx_map": word_idx_map,
                     "vocab": dict(vocab), "max_l": max_l, "n_pre": n_pre}, f)
    print("salvo em", out)


if __name__ == "__main__":
    main()
