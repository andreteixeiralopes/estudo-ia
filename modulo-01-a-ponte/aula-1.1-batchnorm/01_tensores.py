"""
Fundamentos de tensor, construídos passo a passo.
Escrito em NumPy; em PyTorch tudo é igual trocando axis= por dim=.

Rode:  python 01_tensores.py
"""
import numpy as np


def etapa_1_shape():
    """Um tensor é um bloco linear de memória + metadados dizendo como lê-lo.
    O metadado principal é o shape: o tamanho de cada dimensão."""
    print("=" * 60)
    print("ETAPA 1 — shape e eixos")
    print("=" * 60)

    x = np.arange(24).reshape(2, 3, 4)
    print("shape:", x.shape, "| ndim:", x.ndim, "| total de numeros:", x.size)
    print(x)

    # Regra 1: indexar um eixo o REMOVE do shape.
    print("\nx[0]       ->", x[0].shape, " (eixo 0 fixado, sumiu)")
    print("x[0, 2]    ->", x[0, 2].shape, "  (eixos 0 e 1 fixados)")
    print("x[0, 2, 1] -> escalar:", x[0, 2, 1])

    # Regra 2: REDUZIR um eixo (mean, sum) também o remove.
    # Indexar escolhe uma fatia; reduzir funde todas as fatias em uma.


def etapa_2_nchw():
    """No nosso problema o tensor é (N, C, H, W):
    eixo 0 = N imagens, 1 = C canais, 2 = altura, 3 = largura."""
    print("\n" + "=" * 60)
    print("ETAPA 2 — o tensor (N, C, H, W)")
    print("=" * 60)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 4, 4))
    print("x.shape:", x.shape, "->", x.size, "numeros")

    # ':' significa 'todos deste eixo'. Fixando só o canal:
    canal1 = x[:, 1, :, :]
    print("x[:, 1, :, :].shape:", canal1.shape, "->", canal1.size, "numeros")
    print("media desses valores:", round(float(canal1.mean()), 6))
    print("^ este e' o mu_B do canal 1 que o BatchNorm usa")


if __name__ == "__main__":
    etapa_1_shape()
    etapa_2_nchw()
    print("\n[as próximas etapas — redução vetorizada e broadcast — entram aqui]")
