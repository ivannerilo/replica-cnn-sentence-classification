"""CNN de Kim (2014) em PyTorch, espelhando o código Theano original.

Arquitetura: embedding (n x 300) -> convoluções com janelas 3, 4 e 5 (100 mapas cada,
ReLU) -> max-over-time pooling -> dropout 0.5 -> camada softmax (300 -> 2).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("rand", "static", "nonstatic", "multichannel")


class KimCNN(nn.Module):
    def __init__(self, embeddings, variant, n_classes=2, filter_hs=(3, 4, 5),
                 feature_maps=100, dropout=0.5, seed=3435):
        super().__init__()
        assert variant in VARIANTS
        self.variant = variant
        self.dropout = dropout
        emb = torch.from_numpy(np.asarray(embeddings, dtype="float32"))
        k = emb.shape[1]

        # rand / nonstatic: um canal ajustável; static: um canal congelado;
        # multichannel: canal congelado + canal ajustável, ambos iniciados com word2vec.
        self.embed = nn.Embedding.from_pretrained(emb.clone(), freeze=(variant == "static"),
                                                  padding_idx=0)
        self.embed_static = None
        if variant == "multichannel":
            self.embed_static = nn.Embedding.from_pretrained(emb.clone(), freeze=True,
                                                             padding_idx=0)
        in_channels = 2 if variant == "multichannel" else 1

        # Inicialização igual à original para ReLU: W ~ U[-0.01, 0.01], b = 0.
        rng = np.random.RandomState(seed)
        self.convs = nn.ModuleList()
        for h in filter_hs:
            conv = nn.Conv2d(in_channels, feature_maps, kernel_size=(h, k))
            w = rng.uniform(-0.01, 0.01, size=conv.weight.shape).astype("float32")
            conv.weight.data.copy_(torch.from_numpy(w))
            conv.bias.data.zero_()
            self.convs.append(conv)

        # Camada softmax iniciada com zeros, como na LogisticRegression original.
        self.out = nn.Linear(feature_maps * len(filter_hs), n_classes)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def features(self, x):
        e = self.embed(x).unsqueeze(1)  # (B, 1, n, k)
        if self.embed_static is not None:
            e = torch.cat([self.embed_static(x).unsqueeze(1), e], dim=1)  # (B, 2, n, k)
        pooled = [F.relu(conv(e)).squeeze(3).max(dim=2).values for conv in self.convs]
        return torch.cat(pooled, dim=1)  # (B, 300) = camada penúltima z

    def forward(self, x):
        z = self.features(x)
        if self.training:
            # Dropout "clássico" (Hinton et al., 2012), como no original: sem reescalar
            # no treino; no teste os pesos são multiplicados por (1 - p).
            mask = torch.bernoulli(torch.full_like(z, 1 - self.dropout))
            return self.out(z * mask)
        return F.linear(z, self.out.weight * (1 - self.dropout), self.out.bias)

    @torch.no_grad()
    def apply_max_norm(self, s=3.0):
        """Restringe ||w||_2 <= s para o vetor de pesos de cada classe (após cada passo)."""
        w = self.out.weight
        norms = w.norm(dim=1, keepdim=True)
        w.mul_(norms.clamp(max=s) / (1e-7 + norms))

    @torch.no_grad()
    def zero_padding(self):
        """O original zera o vetor do padding (índice 0) após cada atualização."""
        self.embed.weight[0].zero_()
