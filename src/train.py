"""Treino com validação cruzada em 10 folds, espelhando `conv_net_sentence.py` original.

Para cada fold de teste:
  1. completa o treino até um múltiplo de 50 repetindo exemplos aleatórios;
  2. embaralha e separa 10% dos mini-lotes como validação (early stopping);
  3. treina 25 épocas com Adadelta (rho=0.95, eps=1e-6), lotes de 50 embaralhados;
  4. a acurácia de teste reportada é a da época com melhor acurácia de validação.

Uso:
  python src/train.py --variant nonstatic
  python src/train.py --variant rand --folds 0 1 --epochs 5   # teste rápido
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import VARIANTS, KimCNN

ROOT = Path(__file__).resolve().parent.parent
SEED = 3435
FILTER_HS = (3, 4, 5)
BATCH_SIZE = 50
MAX_L = 56  # valor fixo passado pelo código original para a base MR


def sent_to_idx(text, word_idx_map, max_l=MAX_L, filter_h=max(FILTER_HS)):
    """Padding de (filter_h - 1) zeros à esquerda e completa à direita até max_l + 2*pad."""
    pad = filter_h - 1
    x = [0] * pad + [word_idx_map[w] for w in text.split() if w in word_idx_map]
    x += [0] * (max_l + 2 * pad - len(x))
    return x


def make_fold(revs, word_idx_map, fold):
    train, test = [], []
    for r in revs:
        row = sent_to_idx(r["text"], word_idx_map) + [r["y"]]
        (test if r["split"] == fold else train).append(row)
    return np.array(train, dtype="int64"), np.array(test, dtype="int64")


def accuracy(model, x, y, device, bs=1000):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i:i + bs]).to(device)
            pred = model(xb).argmax(1).cpu().numpy()
            correct += (pred == y[i:i + bs]).sum()
    return correct / len(x)


def train_fold(train, test, embeddings, variant, epochs, device, log):
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)

    # Mesmo procedimento do original para completar o lote e separar a validação.
    if len(train) % BATCH_SIZE > 0:
        extra = rng.permutation(train)[:BATCH_SIZE - len(train) % BATCH_SIZE]
        train = np.append(train, extra, axis=0)
    data = rng.permutation(train)
    n_batches = len(data) // BATCH_SIZE
    n_train_batches = int(np.round(n_batches * 0.9))
    tr = data[:n_train_batches * BATCH_SIZE]
    va = data[n_train_batches * BATCH_SIZE:]
    tr_x, tr_y = tr[:, :-1], tr[:, -1]
    va_x, va_y = va[:, :-1], va[:, -1]
    te_x, te_y = test[:, :-1], test[:, -1]

    model = KimCNN(embeddings, variant, filter_hs=FILTER_HS, seed=SEED).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adadelta(params, lr=1.0, rho=0.95, eps=1e-6)

    best_val, test_acc, best_epoch, history = 0.0, 0.0, 0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for b in rng.permutation(n_train_batches):
            sl = slice(b * BATCH_SIZE, (b + 1) * BATCH_SIZE)
            xb = torch.from_numpy(tr_x[sl]).to(device)
            yb = torch.from_numpy(tr_y[sl]).to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.apply_max_norm(3.0)
            model.zero_padding()
        train_acc = accuracy(model, tr_x, tr_y, device)
        val_acc = accuracy(model, va_x, va_y, device)
        if val_acc >= best_val:
            best_val, best_epoch = val_acc, epoch
            test_acc = accuracy(model, te_x, te_y, device)
        history.append({"epoch": epoch, "train": float(train_acc), "val": float(val_acc)})
        log(f"  época {epoch:2d} | {time.time() - t0:5.1f}s | treino {train_acc:.4f} "
            f"| val {val_acc:.4f} | teste@melhor-val {test_acc:.4f}")
    return float(test_acc), best_epoch, history, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--folds", type=int, nargs="*", default=list(range(10)))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="salva os embeddings finais do último fold (para analysis.py)")
    args = parser.parse_args()

    with open(ROOT / "data" / "mr.pkl", "rb") as f:
        d = pickle.load(f)
    embeddings = d["W2"] if args.variant == "rand" else d["W"]
    if embeddings is None:
        raise SystemExit("mr.pkl foi gerado sem word2vec; rode preprocess.py sem --no-w2v.")

    # Cada fold é salvo assim que termina (results/folds/), então uma execução
    # interrompida pode ser retomada: folds já salvos são pulados.
    out_dir = ROOT / "results"
    fold_dir = out_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for fold in args.folds:
        fold_file = fold_dir / f"{args.variant}_fold{fold}.json"
        if fold_file.exists():
            print(f"fold {fold} já concluído ({fold_file.name}); pulando")
            continue
        log_file = open(fold_dir / f"{args.variant}_fold{fold}.log", "w")

        def log(msg):
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()

        log(f"CNN-{args.variant} | dispositivo: {args.device} | épocas: {args.epochs}")
        train, test = make_fold(d["revs"], d["word_idx_map"], fold)
        log(f"fold {fold}: treino {len(train)} | teste {len(test)}")
        acc, best_epoch, history, model = train_fold(train, test, embeddings, args.variant,
                                                     args.epochs, args.device, log)
        log(f"fold {fold}: acurácia de teste {acc:.4f} (melhor época {best_epoch})")
        log_file.close()
        tmp = fold_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fold": fold, "test_acc": acc, "best_epoch": best_epoch,
                                   "history": history}, indent=2))
        tmp.replace(fold_file)
        if args.save_embeddings and fold == max(args.folds):
            np.save(out_dir / f"emb_{args.variant}.npy",
                    model.embed.weight.detach().cpu().numpy())

    results = sorted((json.loads(p.read_text()) for p in fold_dir.glob(f"{args.variant}_fold*.json")),
                     key=lambda r: r["fold"])
    accs = np.array([r["test_acc"] for r in results])
    summary = {"variant": args.variant, "epochs": args.epochs, "mean": float(accs.mean()),
               "std": float(accs.std()), "folds": results}
    print(f"CNN-{args.variant}: média {accs.mean() * 100:.2f}% ± {accs.std() * 100:.2f} "
          f"em {len(accs)} folds")
    with open(out_dir / f"{args.variant}.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
