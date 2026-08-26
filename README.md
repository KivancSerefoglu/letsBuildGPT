# Let's Build GPT from Scratch

This project is a small PyTorch implementation of a character-level language model inspired by Andrej Karpathy's famous tutorial, "Let's build GPT: from scratch, in code, spelled out".

- Tutorial: https://www.youtube.com/watch?v=kCc8FmEb1nY
- Related repo: https://github.com/karpathy/ng-video-lecture

The goal of this project is to learn how a GPT-style transformer works by building it step by step from scratch, using a tiny Shakespeare dataset and training a model to generate text one character at a time.

## Project overview

The repository contains a minimal implementation of:

- a basic bigram language model
- a transformer-based character model similar to GPT
- data loading and training pipeline for text generation
- text generation from a trained model

The code is intentionally educational and easy to follow so it can serve as a study project or as a starting point for deeper experimentation.

## Files in this project

- `bigram.py` — a more complete GPT-style transformer model with:
  - token and positional embeddings
  - multi-head self-attention
  - feed-forward layers
  - transformer blocks
  - training loop and generation loop

- `v2.py` — a simplified version that focuses on the basic bigram mechanism and a smaller training setup.

- `input.txt` — the training dataset, based on the Tiny Shakespeare corpus used in the original tutorial.

- `gpt-dev.ipynb` — notebook version for interactive experimentation and learning.

## What the model does

The model treats text as a sequence of characters. It learns a probabilistic mapping from context to the next character.

At a high level:

1. Encode text into integer token IDs.
2. Split the data into training and validation sets.
3. Build batches of character sequences.
4. Feed those sequences through token embeddings and positional embeddings.
5. Process them through transformer layers with self-attention.
6. Predict the next character.
7. Train using cross-entropy loss.
8. Generate new text autoregressively after training.

## Setup

Make sure Python and PyTorch are installed.

```bash
python3 --version
pip install torch
```

If you are using a virtual environment, activate it before running the scripts.

## Run the project

### Run the transformer version

```bash
python3 bigram.py
```

### Run the simplified version

```bash
python3 v2.py
```

The script will:

- read the dataset from `input.txt`
- train the model for a number of steps
- print training and validation loss periodically
- generate sample text after training

## Expected behavior

Because this is a small educational model trained on a small dataset, the generated text will not be perfect or fully coherent at first. However, it should show learning behavior over time, and the generated output will gradually become more readable as training progresses.

## Learning goals

This project is useful for understanding:

- tokenization and embeddings
- positional encoding
- self-attention
- transformer blocks
- autoregressive generation
- cross-entropy training for language models
- how GPT-like models are built from first principles

## Notes

This repository is intentionally simple and follows the spirit of the original tutorial rather than trying to be a production-ready LLM framework. It is best used as a learning exercise and as a foundation for custom experiments.

## Suggested next steps

- increase the dataset size
- adjust model hyperparameters
- add learning-rate scheduling
- train for more iterations
- compare character-level and word-level models
- explore inference sampling strategies such as temperature and top-k sampling
- extend the model toward a more complete GPT architecture

## License

This project is for educational purposes and is inspired by open-source learning resources. Please check the original tutorial and associated materials for their licensing and usage guidance.
