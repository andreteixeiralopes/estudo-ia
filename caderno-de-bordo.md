# Caderno de bordo

Estado do estudo. Ler este arquivo é o que restaura o contexto numa conversa nova.
Atualizar ao fim de cada aula.

Trilha comentada: https://claude.ai/code/artifact/7a946773-ff56-49bd-a137-58bae23d1ad7

## Estado atual

- **Módulo em curso:** 1 — A ponte 2016→2019
- **Aula atual:** 1.1 — BatchNorm. Teoria e as 3 perguntas fechadas em 27/09/2026.
  Exercício em andamento, sendo construído passo a passo (fundamentos de tensor primeiro).
- **Ponto de partida:** dissertação de mestrado 2016 (UFES), CNN para reconhecimento de
  expressões faciais. Domina backprop, SGD, conv/pooling, augmentation e metodologia de
  validação cruzada entre bases. Lacuna: tudo de 2016 em diante.
- **Python/PyTorch:** dev sênior em outras linguagens, Python iniciante/intermediário,
  sem prática com tensores. As aulas explicam a mecânica de tensor conforme o código avança.

## Protocolo

1. Aula: teoria curta, leitura indicada, 2–3 perguntas, exercício.
2. Andre responde com as próprias palavras e faz o exercício.
3. Correção em registro de orientador exigente — aponta erro conceitual, inclusive pequeno.
4. Registro aqui do que foi feito, dos erros a revisitar e da próxima aula.
5. Módulo 0 (matemática) não é feito em bloco: entra sob demanda, quando uma aula cobrar.

Aula padrão dimensionada para ~1 h.

## Plano do Módulo 1

- [ ] 1.1 BatchNorm — teoria ok, exercício em andamento
- [ ] 1.2 Otimização moderna — momento, Adam, AdamW, agendamento de taxa
- [ ] 1.3 Regularização — dropout, augmentation moderno, early stopping
- [ ] 1.4 Conexões residuais — ResNet e o problema de degradação
- [ ] 1.5 LayerNorm vs BatchNorm — eixos de agregação (retomar o erro da aula 1.1)
- [ ] 1.6 Transfer learning e PyTorch na prática
- [ ] Projeto-âncora: reimplementar a rede da dissertação e comparar com os números de CK+

## Pontos a revisitar

- **Eixos de agregação.** Na aula 1.1 disse que com batch 1 o desvio seria 0 — vale para
  camada densa ou mapa 1x1, não para conv com mapa HxW. Retomar na aula 1.5.
- **Mecanismo vs causa atribuída.** Explicou o BatchNorm pela narrativa do *internal
  covariate shift*, contestada desde 2018. Cobrar a separação nas próximas aulas.
- **Notação.** Apresentar todo símbolo antes de usar.

## Decisões

- 26/09/2026 — Começar pelo Módulo 1 em vez do 0. A matemática que o Módulo 1 cobra já foi
  usada na dissertação; a peça pesada do Módulo 0 (KL, variacional, EDEs) só é cobrada nos
  Módulos 5 e 7, e aprender ferramenta meses antes do problema que a justifica não gruda.
- 30/09/2026 — Repositório no GitHub como registro durável, acessível de qualquer lugar.
