"""Compara nossos resultados com a Tabela 2 do artigo e reproduz a análise da Tabela 3.

Gera `results/comparacao.md` com:
  - acurácia média ± desvio por variante vs. valor do artigo (coluna MR);
  - vizinhos mais próximos (cosseno) de algumas palavras antes e depois do ajuste fino,
    se `train.py --save-embeddings` tiver sido usado para nonstatic/multichannel.

Uso:  python src/analysis.py
"""
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Tabela 2 de Kim (2014), coluna MR (acurácia %, validação cruzada em 10 folds).
PAPER_MR = {"rand": 76.1, "static": 81.0, "nonstatic": 81.5, "multichannel": 81.1}
PROBE_WORDS = ["bad", "good", "n't", "!", ","]


def results_table():
    lines = ["| Modelo | Artigo (MR) | Réplica (média ± desvio) | Diferença (p.p.) | Folds |",
             "|---|---|---|---|---|"]
    for variant, paper in PAPER_MR.items():
        path = RESULTS / f"{variant}.json"
        if not path.exists():
            lines.append(f"| CNN-{variant} | {paper:.1f} | (não executado) | – | – |")
            continue
        r = json.loads(path.read_text())
        mean, std = r["mean"] * 100, r["std"] * 100
        lines.append(f"| CNN-{variant} | {paper:.1f} | {mean:.2f} ± {std:.2f} | "
                     f"{mean - paper:+.2f} | {len(r['folds'])} |")
    return "\n".join(lines)


def neighbors(emb, idx, inv, word, k=4):
    v = emb[idx[word]]
    sims = emb @ v / (np.linalg.norm(emb, axis=1) * np.linalg.norm(v) + 1e-8)
    sims[0] = sims[idx[word]] = -np.inf  # ignora padding e a própria palavra
    return [inv[i] for i in np.argsort(-sims)[:k]]


def neighbors_table():
    tuned_path = next((RESULTS / f"emb_{v}.npy" for v in ("multichannel", "nonstatic")
                       if (RESULTS / f"emb_{v}.npy").exists()), None)
    if tuned_path is None:
        return "_Rode `train.py --save-embeddings` (nonstatic ou multichannel) para gerar._"
    with open(ROOT / "data" / "mr.pkl", "rb") as f:
        d = pickle.load(f)
    idx = d["word_idx_map"]
    inv = {i: w for w, i in idx.items()}
    static, tuned = d["W"], np.load(tuned_path)
    lines = [f"Embeddings ajustados: `{tuned_path.name}`", "",
             "| Palavra | Vizinhos no word2vec (estático) | Vizinhos após ajuste fino |",
             "|---|---|---|"]
    for w in PROBE_WORDS:
        if w in idx:
            lines.append(f"| `{w}` | {', '.join(neighbors(static, idx, inv, w))} | "
                         f"{', '.join(neighbors(tuned, idx, inv, w))} |")
    return "\n".join(lines)


def main():
    md = ("# Comparação com o artigo\n\n## Acurácia na base MR\n\n" + results_table()
          + "\n\n## Vizinhos mais próximos (análise da Tabela 3)\n\n" + neighbors_table() + "\n")
    (RESULTS / "comparacao.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
