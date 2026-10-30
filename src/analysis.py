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
    plot_figures()


def plot_figures():
    """Acurácia por fold vs. artigo e curvas de treino/validação do fold 0."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib não instalado; figuras não geradas")
        return
    runs = {v: json.loads((RESULTS / f"{v}.json").read_text())
            for v in PAPER_MR if (RESULTS / f"{v}.json").exists()}
    if not runs:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (v, r) in enumerate(runs.items()):
        accs = [f["test_acc"] * 100 for f in r["folds"]]
        ax.scatter([i] * len(accs), accs, color="#3D6FD9", alpha=0.5, label="fold" if i == 0 else None)
        ax.hlines(r["mean"] * 100, i - 0.25, i + 0.25, color="#121A2B", lw=2,
                  label="média da réplica" if i == 0 else None)
        ax.scatter(i, PAPER_MR[v], marker="D", color="#E07A2F", zorder=3,
                   label="artigo" if i == 0 else None)
    ax.set_xticks(range(len(runs)), [f"CNN-{v}" for v in runs])
    ax.set_ylabel("acurácia de teste (%)")
    ax.set_title("Base MR: acurácia por fold vs. artigo")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "acuracia_por_fold.png", dpi=150)

    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 3.2), sharey=True)
    for ax, (v, r) in zip(np.atleast_1d(axes), runs.items()):
        h = r["folds"][0]["history"]
        ax.plot([e["epoch"] for e in h], [e["train"] for e in h], color="#3D6FD9", label="treino")
        ax.plot([e["epoch"] for e in h], [e["val"] for e in h], color="#E07A2F", label="validação")
        ax.set_title(f"CNN-{v} (fold {r['folds'][0]['fold']})")
        ax.set_xlabel("época")
    np.atleast_1d(axes)[0].set_ylabel("acurácia")
    np.atleast_1d(axes)[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "curvas_treino.png", dpi=150)
    print("figuras salvas em results/")


if __name__ == "__main__":
    main()
