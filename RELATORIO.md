# Relatório de Réplica — Kim (2014)

> *Convolutional Neural Networks for Sentence Classification*, Yoon Kim, EMNLP 2014.
> https://arxiv.org/abs/1408.5882

## 1. Objetivo

Reproduzir a Tabela 2 do artigo na base **MR** para as quatro variantes da CNN
(rand, static, non-static e multichannel), usando os mesmos dados, pré-processamento,
arquitetura, hiperparâmetros e protocolo de avaliação, e discutir as divergências.

## 2. Dados

| | Artigo (Tabela 1) | Réplica |
|---|---|---|
| Frases (N) | 10662 | 10662 |
| Classes (c) | 2 | 2 |
| Tamanho médio (l) | 20 | 20,4 |
| Vocabulário \|V\| | 18765 | 18765 |
| Palavras no word2vec \|V_pre\| | 16448 | 16448 |
| Teste | CV (10 folds) | CV (10 folds) |

As estatísticas batem **exatamente**, o que indica que a tokenização foi reproduzida fielmente.
Fonte: `rt-polarity.pos/.neg` (Pang & Lee, 2005), obtidos do repositório oficial do autor
(idênticos aos da Cornell). Vetores: word2vec Google News, 300 dimensões.

## 3. Método reproduzido

Seguimos o artigo e, onde ele é omisso, o código oficial do autor
(https://github.com/yoonkim/CNN_sentence, em Theano):

- **Pré-processamento** (`src/preprocess.py`): a mesma função `clean_str` (remove caracteres
  fora de `[A-Za-z0-9(),!?'\`]`, separa contrações e pontuação, converte para minúsculas).
  Cada frase recebe um fold aleatório de 0 a 9.
- **Embeddings**: palavras fora do word2vec são iniciadas com U[−0,25; 0,25]. O CNN-rand
  usa U[−0,25; 0,25] para todas as palavras. O índice 0 (padding) é um vetor zero, mantido
  zerado após cada atualização.
- **Entrada**: 4 zeros de padding à esquerda e completada à direita até 56 + 8 = 64 posições.
- **Modelo** (`src/model.py`): convoluções com janelas 3, 4 e 5 (100 mapas cada, ReLU),
  max-over-time pooling, dropout 0,5 e softmax. Inicialização do original: filtros
  U[−0,01; 0,01], camada de saída com zeros.
- **Regularização**: dropout "clássico" (pesos multiplicados por 0,5 no teste) e restrição
  ‖w‖₂ ≤ 3 nos pesos da camada de saída após cada passo.
- **Treino** (`src/train.py`): Adadelta (ρ = 0,95, ε = 1e-6), mini-lotes de 50 em ordem
  aleatória, 25 épocas. 10% do treino vira validação. A acurácia de teste reportada é a da
  época com melhor validação. Média sobre os 10 folds.

## 4. Resultados

<!-- Preencher com results/comparacao.md após rodar todas as variantes. -->

| Modelo | Artigo (MR) | Réplica (média ± desvio, 10 folds) | Diferença |
|---|---|---|---|
| CNN-rand | 76,1 | | |
| CNN-static | 81,0 | | |
| CNN-non-static | 81,5 | | |
| CNN-multichannel | 81,1 | | |

## 5. Diferenças de implementação conhecidas

1. **Framework**: Theano (2014, float64 em partes do pipeline) → PyTorch (float32).
   Somas em ponto flutuante e geradores aleatórios diferentes mudam os resultados em
   frações de ponto percentual.
2. **Sorteio dos folds**: o original sorteia os folds sem semente fixa, então nossa partição
   não é a mesma do autor. Mesma distribuição, frases diferentes em cada fold.
3. **Multichannel**: o código público do autor **não implementa** essa variante. O artigo
   diz que "cada filtro é aplicado aos dois canais e os resultados são somados". Usamos uma
   convolução com 2 canais de entrada, em que cada canal tem pesos próprios e as respostas
   são somadas. A leitura alternativa (o mesmo filtro nos dois canais) é igualmente
   compatível com o texto.
4. **Não determinismo em GPU**: algumas operações do cuDNN não são determinísticas, então
   rodar duas vezes no Colab pode dar números ligeiramente diferentes.

## 6. Dificuldades

- O servidor da Cornell bloqueou o download no nosso ambiente. Usamos o espelho no
  repositório oficial do autor, com os mesmos arquivos.
- O word2vec do Google News tem cerca de 1,6 GB compactado e precisa de uns 4 GB de RAM para
  carregar. Por isso filtramos só as palavras do vocabulário e guardamos a matriz em `data/mr.pkl`.
- O código original é Python 2 + Theano, que não roda mais em ambientes atuais. Foi preciso
  traduzir cada detalhe (dropout sem reescala, max-norm, padding, completar o último lote,
  separação da validação) para PyTorch.
- Em CPU, cada variante leva mais de 1 hora (10 folds × 25 épocas). Na GPU do Colab, poucos minutos.

## 7. Análise crítica

<!-- Completar após os resultados. Pontos a discutir: -->

- A réplica confirma a **conclusão principal**? (word2vec pré-treinado ≫ aleatório; ganho de
  cerca de 5 p.p.)
- O ganho do ajuste fino (non-static vs. static) é pequeno no artigo (+0,5). Ele está dentro
  do desvio entre folds?
- A multichannel não supera a non-static no MR, como o próprio autor observa (resultados "mistos").
- Vizinhos de palavras antes e depois do ajuste fino (`results/comparacao.md`, análise da
  Tabela 3): `good` deixa de ser vizinho de `bad`?
- Limitações do artigo: só uma execução por configuração, sem desvio padrão nem teste de
  significância. Diferenças de 0,5 p.p. entre variantes provavelmente não são significativas.
