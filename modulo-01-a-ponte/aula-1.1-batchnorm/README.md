# Aula 1.1 — Normalização interna: BatchNorm

## O problema

Na dissertação de 2016, a normalização de intensidade era aplicada às imagens antes da
rede. A justificativa era correta: imagens da mesma expressão variam em brilho e
contraste, e essa variação é ruído que a rede teria de aprender a ignorar.

O que essa solução tem de limitado: ela age **uma vez**, **na entrada**, **antes do treino**.
O mesmo problema existe dentro da rede — a entrada da camada 5 é a saída da camada 4, e a
camada 4 muda a cada passo de gradiente. Do ponto de vista da camada 5, a distribuição das
suas próprias entradas se mexe durante o treino inteiro.

Ioffe e Szegedy (2015) propuseram normalizar também lá dentro, a cada camada, a cada passo.

## A mecânica

Para um mini-batch B, em cada canal, calcula-se média e variância sobre o batch e sobre as
posições espaciais:

    x_hat = (x - mu_B) / sqrt(var_B + eps)
    y     = gamma * x_hat + beta

com gamma e beta aprendidos por backpropagation, um par por canal.

### Por que gamma e beta existem

Normalizar à força tem custo. Com entrada sempre em média 0 e variância 1, uma sigmoide
opera só na sua faixa central — a parte quase reta. Uma pilha de camadas lineares equivale
a uma única camada linear: o ganho de profundidade desapareceria. Com ReLU, o corte fica
travado em aproximadamente 50% das ativações, independentemente do que a camada precisa.

gamma e beta devolvem essas liberdades. E há o argumento formal: se gamma = sigma_B e
beta = mu_B, a transformação é exatamente a identidade. Logo o BatchNorm **não remove
nenhuma função do repertório da rede** — é uma reparametrização, não uma restrição.

O ganho está no caminho, não no destino: a escala de cada canal deixa de depender da
coordenação delicada de centenas de pesos e vira dois parâmetros dedicados, ajustáveis
diretamente pelo gradiente.

Consequência prática: como beta já é um deslocamento aprendido, o viés da convolução
anterior fica redundante — daí `nn.Conv2d(..., bias=False)` antes de um BatchNorm.

### Treino e inferência são assimétricos

No treino usam-se as estatísticas do batch. Na inferência não há batch, então a camada usa
uma média móvel acumulada durante o treino:

    running_mean = (1 - m) * running_mean + m * mu_B
    running_var  = (1 - m) * running_var  + m * var_B_nao_enviesada

com m = 0.1 por padrão em PyTorch. Em PyTorch esses valores são *buffers*: aparecem no
`state_dict` mas não em `parameters()` — o otimizador não os toca.

Detalhe que morde na implementação: a normalização no treino usa a variância **enviesada**
(divide por N), enquanto o `running_var` acumula a **não enviesada** (divide por N-1).

### A explicação oficial provavelmente está errada

O paper de 2015 atribui o funcionamento ao *internal covariate shift*. Santurkar et al.
(2018) injetaram ruído deliberado após a normalização, recriando artificialmente esse
deslocamento, e o treino continuou bom. O efeito real parece ser outro: o BatchNorm
suaviza a superfície de perda, estabilizando o gradiente.

Estabilizar a distribuição é o que o método **faz**. Não é por isso que ele **funciona**.

## Leitura

- Obrigatória: Ioffe & Szegedy, *Batch Normalization* (2015), seções 1 a 3.4 — https://arxiv.org/abs/1502.03167
- Recomendada: Santurkar et al., *How Does Batch Normalization Help Optimization?* (2018),
  introdução e seção 2 — https://arxiv.org/abs/1805.11604
