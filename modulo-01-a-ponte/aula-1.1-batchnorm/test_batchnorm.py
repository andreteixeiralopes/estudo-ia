"""
Verificação: a implementação de batchnorm.py precisa bater com torch.nn.BatchNorm2d,
na saída E no estado acumulado, nos dois modos.

Rode:  python test_batchnorm.py
"""
import torch
from batchnorm import meu_batchnorm

ATOL = 1e-5


def main():
    torch.manual_seed(0)
    N, C, H, W = 8, 3, 5, 5

    bn = torch.nn.BatchNorm2d(C)
    gamma = torch.ones(C)
    beta = torch.zeros(C)
    running_mean = torch.zeros(C)
    running_var = torch.ones(C)

    # --- modo treino: alguns passos, comparando a cada um -------------------
    bn.train()
    for passo in range(3):
        x = torch.randn(N, C, H, W)

        y_ref = bn(x)
        y_meu, running_mean, running_var = meu_batchnorm(
            x, gamma, beta, running_mean, running_var, training=True
        )

        checa(f"treino passo {passo} — saida", y_meu, y_ref)
        checa(f"treino passo {passo} — running_mean", running_mean, bn.running_mean)
        checa(f"treino passo {passo} — running_var", running_var, bn.running_var)

    # --- modo inferência ----------------------------------------------------
    bn.eval()
    x = torch.randn(N, C, H, W)
    y_ref = bn(x)
    y_meu, _, _ = meu_batchnorm(
        x, gamma, beta, running_mean, running_var, training=False
    )
    checa("eval — saida", y_meu, y_ref)

    print("\nTudo bateu dentro de", ATOL)


def checa(nome, meu, ref):
    if meu is ... or meu is None:
        raise SystemExit(f"[{nome}] ainda nao implementado")
    diff = (meu - ref).abs().max().item()
    status = "ok " if diff < ATOL else "FALHOU"
    print(f"[{status}] {nome:38s} diferenca maxima = {diff:.3e}")
    if diff >= ATOL:
        raise SystemExit(
            "\nDica: diferenca da ordem de 1e-1 ou maior costuma ser eixo errado;\n"
            "diferenca pequena mas acima da tolerancia costuma ser variancia\n"
            "enviesada x nao enviesada."
        )


if __name__ == "__main__":
    main()
