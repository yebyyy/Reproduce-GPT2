# Reproduce-GPT2
This repo is for rebuilding the OpenAi **GPT2(124M)** model.

As a Decoder-only Transformer model, the structure of GPT follows the decoder structure as in the [Attention is All You Need Paper](https://arxiv.org/pdf/1706.03762).

The original OpenAI Blog Post can be found at [Better language models and their implications](https://openai.com/index/better-language-models/), which links to the paper [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and the github repo [gpt2](https://github.com/openai/gpt-2).

Besides the **GPT2** paper, this repo also references the **GPT3** paper, which is the [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165).

The model training, optimization, and hyperparameter tuning follows both papers and implements **Flash Attention** as mentioned in both [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135) and [FlashAttention-2:Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/pdf/2307.08691).

The Dataset used to train the model is the 10BT sample from [Huggingface Fineweb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu?row=8).

The model evaluation uses the [Hellaswag LLM Benchmark](https://arxiv.org/pdf/1905.07830), and achieved , which is .

Since this repo uses `Pytorch`, the [huggingface GPT2 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) is also referenced.
