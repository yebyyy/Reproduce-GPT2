# Reproduce-GPT2
This repo is for rebuilding the openai GPT2(124M) model.

As a Decoder-only Transformer model, the structure of GPT follows the decoder structure as in the [Attention is All You Need Paper](https://arxiv.org/pdf/1706.03762).

The original OpenAI Blog Post can be found at [Better language models and their implications](https://openai.com/index/better-language-models/), which links to the paper [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and the github repo [gpt2](https://github.com/openai/gpt-2).

Besides the gpt2 paper, this repo also references the gpt3 paper, which is the [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165).

Since this repo uses `Pytorch`, the [huggingface GPT2 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) is also referenced.
