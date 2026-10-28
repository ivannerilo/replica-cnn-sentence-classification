"""Baixa a base MR (Movie Review, Pang & Lee 2005) usada no artigo.

Tenta primeiro o espelho no repositório oficial do autor (yoonkim/CNN_sentence),
que contém exatamente os arquivos rt-polarity.pos/.neg usados no artigo; se falhar,
baixa o tarball original da Cornell.

Uso:  python data/download_data.py
"""
import io
import tarfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
FILES = ["rt-polarity.pos", "rt-polarity.neg"]
MIRROR = "https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/{}"
CORNELL = "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"


def from_mirror():
    for name in FILES:
        with urllib.request.urlopen(MIRROR.format(name), timeout=60) as r:
            (DATA_DIR / name).write_bytes(r.read())


def from_cornell():
    with urllib.request.urlopen(CORNELL, timeout=60) as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    for member in tar.getmembers():
        name = Path(member.name).name
        if name in FILES:
            (DATA_DIR / name).write_bytes(tar.extractfile(member).read())


def main():
    if all((DATA_DIR / f).exists() for f in FILES):
        print("Base MR já existe em", DATA_DIR)
        return
    try:
        from_mirror()
    except Exception as e:  # noqa: BLE001
        print("Espelho do GitHub falhou (%s); tentando Cornell..." % e)
        from_cornell()
    for f in FILES:
        n = sum(1 for _ in open(DATA_DIR / f, "rb"))
        print(f"{f}: {n} frases")


if __name__ == "__main__":
    main()
