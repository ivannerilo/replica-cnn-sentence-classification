# Réplica de Artigo — CNN para Classificação de Sentenças

Trabalho prático da disciplina de Inteligência Artificial (Aula 11 — Machine Learning):
reproduzir os resultados de um artigo científico que usa redes neurais.

**Artigo replicado:** Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification.*
EMNLP 2014. https://arxiv.org/abs/1408.5882

**Integrantes:**
- Alexandre Gustavo Weiss
- Alexandre Luft Duarte
- Douglas Gabriel Sierota
- Ivan Lopes Nerilo
- José Augusto Henn
- Josué Henrique Moreira Borges

## O que o artigo faz

Uma CNN de uma única camada convolucional sobre vetores de palavras (word2vec) classifica
sentenças. O artigo compara quatro variantes:

| Variante | Vetores de palavras |
|---|---|
| CNN-rand | aleatórios, aprendidos no treino |
| CNN-static | word2vec, congelados |
| CNN-non-static | word2vec, ajustados (fine-tuning) |
| CNN-multichannel | dois canais word2vec: um congelado e um ajustado |

Replicamos a coluna **MR** (críticas de cinema, positivo/negativo, 10.662 frases, validação
cruzada em 10 folds) da Tabela 2.

## Estrutura

```
data/download_data.py      baixa a base MR (rt-polarity.pos/.neg)
src/preprocess.py          tokenização, folds e matrizes de embeddings (fiel ao original)
src/model.py               CNN em PyTorch (inicialização, dropout e max-norm do original)
src/train.py               validação cruzada 10 folds com early stopping na validação
src/analysis.py            tabela comparativa com o artigo + vizinhos de palavras (Tabela 3)
notebooks/                 notebook pronto para o Google Colab
results/                   logs, JSON por variante e comparacao.md
RELATORIO.md               processo, resultados e análise crítica
```

## Como rodar

### Google Colab (recomendado)

Abra `notebooks/replica_kim2014.ipynb` no Colab, ative a GPU e execute todas as células.

### Local

```bash
pip install -r requirements.txt
python data/download_data.py
python src/preprocess.py                 # baixa o word2vec (~1,6 GB) via gensim
python src/train.py --variant rand
python src/train.py --variant static
python src/train.py --variant nonstatic --save-embeddings
python src/train.py --variant multichannel --save-embeddings
python src/analysis.py
```

Se já tiver o arquivo do word2vec: `python src/preprocess.py --w2v GoogleNews-vectors-negative300.bin.gz`.

## Hiperparâmetros (Seção 3.1 do artigo)

Janelas 3, 4 e 5 com 100 mapas cada, ReLU, dropout 0,5, restrição de norma L2 s = 3,
mini-lotes de 50, Adadelta (ρ = 0,95, ε = 1e-6), 25 épocas, 10% do treino como validação
e early stopping pela acurácia de validação.

## Resultados

Ver `results/comparacao.md` e a análise em `RELATORIO.md`.
