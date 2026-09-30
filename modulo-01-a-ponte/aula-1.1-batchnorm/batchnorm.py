"""
Exercício da aula 1.1 — implementar o forward do BatchNorm à mão.

Especificação
-------------
Entrada : x de forma (N, C, H, W)
Saída   : mesma forma

Por canal (vetores de tamanho C):
    gamma, beta                 parâmetros      (init 1 e 0)
    running_mean, running_var   estado          (init 0 e 1)

TREINO
  1. mu e var do batch, por canal, agregando sobre N, H, W. Variância ENVIESADA.
  2. x_hat = (x - mu) / sqrt(var + eps),  eps = 1e-5
  3. y = gamma * x_hat + beta
  4. atualizar o estado, usando a variância NÃO ENVIESADA só aqui:
       running_mean = (1 - m) * running_mean + m * mu
       running_var  = (1 - m) * running_var  + m * var_nao_enviesada
     com m = 0.1

INFERÊNCIA
  usar running_mean e running_var; não atualizar nada.

Onde quase todo mundo tropeça (de propósito)
--------------------------------------------
- Os eixos: mu precisa ter forma (C,) — um número por canal. Quais dimensões reduzir?
- O broadcast: gamma tem forma (C,) e x_hat tem (N, C, H, W). Multiplicar direto não
  alinha as dimensões. gamma e beta precisam ser remodelados.
"""
import torch


def meu_batchnorm(x, gamma, beta, running_mean, running_var,
                  training=True, eps=1e-5, momentum=0.1):
    """Devolve (y, running_mean, running_var)."""

    if training:
        mu = ...          # TODO: sobre quais dims?  resultado deve ter shape (C,)
        var = ...         # TODO: idem, enviesada (unbiased=False)

        x_hat = ...       # TODO

        # TODO: atualizar running_mean e running_var
        # cuidado: aqui a variância é a NÃO enviesada (unbiased=True)

    else:
        x_hat = ...       # TODO: usar o estado acumulado

    y = ...               # TODO: gamma * x_hat + beta, com o reshape certo
    return y, running_mean, running_var
