# A trilha — 12 módulos

Versão viva e comentada: https://claude.ai/code/artifact/7a946773-ff56-49bd-a137-58bae23d1ad7

Cada módulo traz: o que da formação de 2016 ainda serve, o material estruturado,
os papers originais em ordem de leitura e um critério de conclusão verificável.

| # | Módulo | Núcleo |
| --- | --- | --- |
| 0 | Reafiação matemática | Álgebra linear, probabilidade, KL, otimização. Não feito em bloco: entra sob demanda |
| 1 | A ponte 2016→2019 | Dropout, BatchNorm, Adam/AdamW, ResNet, LayerNorm, transfer learning, PyTorch |
| 2 | Sequências e atenção | seq2seq, Bahdanau, *Attention Is All You Need*, BERT/GPT-2, scaling laws |
| 3 | LLMs por dentro | CS336 de Stanford; tokenização, RoPE, RMSNorm, GQA, SwiGLU, MoE |
| 4 | Pré-treino em escala | Chinchilla, dados, paralelismo, precisão numérica, relatórios técnicos |
| 5 | Pós-treino e raciocínio | SFT, RLHF, DPO, RLVR/GRPO, modelos de raciocínio |
| 6 | Inferência e eficiência | KV cache, quantização, decodificação especulativa, LoRA, vLLM, contexto longo |
| 7 | Modelos generativos | VAE, GAN, DDPM, score-based, latent diffusion, flow matching |
| 8 | Visão moderna e multimodal | ViT, CLIP, DINOv2, SAM, VLMs — fecha o ciclo com a dissertação |
| 9 | Agentes e avaliação | ReAct, tool use, RAG, RL agêntico, construção de evals privados |
| 10 | Interpretabilidade e alinhamento | Circuits, superposição, autoencoders esparsos |
| 11 | A fronteira | Questões em aberto e como acompanhá-las |

## Projetos-âncora

- [ ] M1–2 — reimplementar a rede da dissertação em PyTorch e comparar com os números de CK+
- [ ] M3 — treinar um GPT pequeno do zero, tokenizador próprio, corpus em português
- [ ] M4–5 — fine-tune com LoRA e depois DPO sobre pares rotulados pelo próprio autor
- [ ] M6 — servir o modelo quantizado com vLLM e medir latência e vazão
- [ ] M7 — DDPM e flow matching do zero sobre um conjunto de rostos
- [ ] M8 — reavaliar CK+/JAFFE com CLIP zero-shot contra os resultados de 2016
- [ ] M9 — construir um conjunto de avaliação privado do próprio domínio
- [ ] M10 — autoencoder esparso sobre as ativações do GPT pequeno
