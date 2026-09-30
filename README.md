# Trilha de ML — de 2016 à fronteira

Retomada de estudo em machine learning, partindo da dissertação de mestrado de 2016
(*Facial Expression Recognition using Deep Learning — Convolutional Neural Network*, UFES)
até o estado da arte atual.

Formato: aulas conduzidas em conversa com o Claude — teoria curta, leitura indicada,
perguntas de verificação e exercício de implementação. Este repositório é o registro
durável: o que foi estudado, o que foi respondido, o que foi corrigido e o código escrito.

## Estrutura

```
caderno-de-bordo.md          estado atual do estudo — ler primeiro
trilha.md                    os 12 módulos, do zero à fronteira
modulo-01-a-ponte/           2016 → 2019: o que virou padrão logo após a dissertação
  aula-1.1-batchnorm/
    README.md                teoria da aula
    perguntas.md             perguntas, respostas dadas e correções
    01_tensores.py           fundamentos de tensor: shape, eixos, redução, broadcast
    batchnorm.py             o exercício (esqueleto a completar)
    test_batchnorm.py        verificação contra torch.nn.BatchNorm2d
```

## Ambiente

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CPU basta para todo o Módulo 1.

## Como retomar depois de um tempo parado

1. Ler `caderno-de-bordo.md` — diz exatamente onde o estudo parou.
2. Abrir uma conversa nova com o Claude, colar o conteúdo do caderno, seguir dali.

## Convenção de commits

Um commit por aula concluída, mensagem `aula X.Y: <assunto>`. Exercícios resolvidos
entram em commit próprio, `exercicio X.Y: <assunto>`, para que o histórico mostre a
evolução e não só o resultado final.
