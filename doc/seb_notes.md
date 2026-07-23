# Goal
This project aims to reproduce the paper "[Physics of Language Models: Part 2.1](https://physics.allen-zhu.com/part-2-grade-school-math/part-2-1)".

## I want to investigate:
🔄 = reproduction  
↪️ = partial extension  
↗️ = full extension

1. 🔄 Do models perform better if the question is asked at the start or at the end of the fact hints?
2. 🔄 What does the model know?  
  ➜ 🔄 Use probes to predict `value(A)`, `can_next(A)`.
3. ↪️ How does depth vs. width affect performance of transformers on reasoning tasks?  
  ➜ ↗️ Calculate correlation with the reasoning task performance of
    * ↗️ amount of training compute (FLOPs)
    * ↪️ number of model parameters
    * ↪️ number of transformer layers
    * ↗️ width of MLP layers in transformer
    * ↗️ width of residual stream in transformer
    * ↪️ number of attention heads
4. ↗️ What is the difference between linear probes and V-probes? Do we get different results depending on which one we use?

## What I do NOT want to replicate:
* Dataset generation ➔ use the author's dataset (iGSM)/ code to generate a new dataset

# New vs. reused code:
## new (for personal learning):
* implement linear probes
* implement V-probes
* statistical analysis (correlation)
* visualisations
* training code for transformers
* test framework

most of this has to be written from scratch as the original code is not available.

## reused code:
* dataset generation (iGSM)
* common transformer code (from HuggingFace)


# Expected Learning Outcomes
* Implement linear probes as simple mech. interp. technique that requires low-level modification of the model architecture.  
  ➜ solidify understanding of components, gain confidence that I can do this type of modification
* Evaluate depth vs. width experiments for transformers on reasoning tasks.
  ➜ better understanding of hyperparameters and thus challenges and trends in LLM research

## TODO
Q: Is the model actually trained on problem statement tokens? Do those get gradient updates?
Possibly only trained on solution tokens?
