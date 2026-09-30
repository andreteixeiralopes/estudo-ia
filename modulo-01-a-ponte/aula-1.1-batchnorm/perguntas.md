# Aula 1.1 — perguntas, respostas e correções

## 1. Se a intenção é normalizar, por que existem gamma e beta?

**Não respondida — pedi a explicação.** Consta no README da aula.

Resumo: sem gamma e beta a normalização seria uma restrição que encolhe o conjunto de
funções representáveis (sigmoide presa na região linear, ReLU travada em ~50% de corte).
Com eles, a identidade é recuperável (gamma = sigma_B, beta = mu_B), então nada se perde
em expressividade — muda só a parametrização, e ela fica muito mais fácil de otimizar.

**Pergunta de acompanhamento:** se a identidade é recuperável, por que a rede treinada
quase nunca a escolhe? Porque ela está *disponível*, não é *desejável*. O gradiente
empurra gamma e beta para o que reduz a perda. Se coincidissem sistematicamente com a
identidade, a conclusão honesta seria que o BatchNorm não serve para nada naquela rede.

## 2. O que o BatchNorm consegue que a normalização de intensidade de 2016 não conseguiria?

**Resposta dada:** "a minha normalização era aplicada antes de qualquer processamento da
rede, batchnorm acontece após cada camada. batchnorm ajuda a camada posterior, durante o
treinamento, pois ajusta as entradas para um mesmo padrão."

**Correção.** A diferença de *lugar* está certa e é metade da resposta. Faltaram duas:

- **Estático vs adaptativo.** A normalização de 2016 era calculada uma vez e congelada. O
  BatchNorm recalcula a cada passo porque o que ele normaliza não para quieto — os pesos da
  camada anterior acabaram de mudar. Normalização estática contra alvo móvel é uma
  impossibilidade, não uma limitação de projeto: a estatística necessária nem existe antes
  do treino começar.
- **Escolhido vs aprendido.** A transformação de 2016 foi definida pelo pesquisador; gamma e
  beta são aprendidos, canal a canal, pelo gradiente.

**Cobrança.** "Ajusta as entradas para um mesmo padrão" é a narrativa do *internal covariate
shift*, contestada por Santurkar (2018) e discutida na própria aula. O que o método faz não
é a razão pela qual ele funciona.

## 3. Rede treina bem, mas em produção, classificando uma imagem por vez, o desempenho cai.

**Resposta dada:** "média será igual a imagem e desvio será 0."

**Correção.** Certo no caso particular, errado no geral.

- **Camada densa, ou mapa já reduzido a 1x1:** existe um único valor por canal. Variância 0,
  x_hat = 0, saída = beta para qualquer entrada. Colapso total. A resposta vale aqui.
- **Camada convolucional com mapa HxW > 1x1:** com N=1 ainda restam H*W valores por canal.
  A média é espacial, o desvio **não** é zero. Cada imagem passa a ser normalizada pelas
  próprias estatísticas em vez das da população — a rede perde a capacidade de comparar
  imagens entre si. Acurácia cai muito sem zerar, e a falha é atribuída a overfitting.

Isso tem nome: é *Instance Normalization* (Ulyanov, 2016), usada de propósito em
transferência de estilo justamente por apagar contraste e intensidade globais de cada
imagem. Rodar BatchNorm em modo treino com batch 1 converte, sem querer, uma na outra.

**Diagnóstico em dois minutos:** passar a mesma imagem por `model.train()` e `model.eval()`
e comparar. Sinal definitivo: variar o tamanho do batch de teste mudar a acurácia — um
modelo correto é indiferente a isso na inferência.

## Pontos a revisitar

- **Eixos de agregação.** Retomar na aula 1.5, que é exatamente sobre isso: o eixo varrido
  é a única coisa que separa BatchNorm, LayerNorm, InstanceNorm e GroupNorm.
- **Mecanismo vs causa atribuída.** Cobrar a separação nas próximas aulas.
