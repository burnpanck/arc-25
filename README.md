# My approach to ARC Prize 2025

This repo contains all the code developed for my attempts at solving
the ARC AGI 2 challenges within the ARC Prize 2025 format.
It has multiple loosely coupled components supporting the two strategies
at reaching a single goal: AGI as measured by the ARC Prize 2025.

Keywords:
MAE,
LLM,
ViT,
JAX,
Perceiver,
Focal loss,
MHA/GQA,
Dihedral symmetry equivariance,
Latent program encoder-decoder,
Few-shot learning,
Gradient accumulation,
Vertex AI.


## Strategies

The main proposition underyling all of these approaches are the apparent
trouble current AI systems have in properly identifying underlying semantic
objects to which the "core knowledge" is then applied.
This is corrobrated by manual experimentation: If we give a natural language
description of the semantic objects in the input/output pairs to a SOTA
LLM ChatBot, it is able to identify the underlying rule without problems.

The solution therefore is to improve the semantic understanding of the tasks,
utilising a domain-adapted semantic encoder specifically for these tasks.

From there, we pursue two separate avenues to validate this propositon:
A *transductive* and an *inductive* approaches to solve the ARC AGI challenges.

### *Transductive* approach: Encoder-decoder driven by a "latent program"

Architecture:
 - Encoder: Distills semantic meaning from raw input grids.
   Output are tokens encoding that semantic meaning.
 - Latent program: Embedding of the underlying rule.
 - Decoder: Combines input semantics together with the rule to
   predict output grids.

At **inference** time, the latent program could be identified using:
 1. Few-shot learning: Backpropagation from the given I/O pairs.
 2. Direct prediction: Apply the encoder to both the input and output
    grids of an example, then train a separate rule identifier network.
 3. A combination of the two: Inititalise latent program from an approximate
    prediction, then backpropagate.

During **training,** we simultaneously train the decoder and the latent program
on a large dataset of I/O pairs ([Re-ARC]), while keeping the encoder frozen.

**Currently implemented**:
 - *Few-shot learning:* Backpropagation using standard gradient descent.
 - *Test-time scaling:* Multiple latent programs are learned from different
   random initialisations; the best two are picked to make predictions.
 - *Efficient training:*
   - Data-parallel training on multiple accelerators.
   - Bucket grids into similar sizes to avoid padding overhead.
   - Gradient accumulation over multiple minibatches.
   - Training on Vertex AI infrastructure.

### *Inductive* approach: LLM-Agent

The LLM-Agent builds and iteratively refines an explicit *Rule* as expressed in a *python-based DSL*.

Architecture:
 - Custom VLM combining a domain-specific encoder
   with a fine-tuned open-weights LLM (Gemma 3).
 - CoT reasoning to formulate rule hypotheses.
 - RAG over DSL documentation and example solution
   to help with DSL code generation.
 - Tool use / function calling:
   - Specialsed "focus"-tool allowing LLM to look at
     individual aspects of the input/output grid.
   - Normal python executor to help with quantitative reasoning
   - Reference search (internal database) to augment RAG
 - ReAct style iterative refinement after
   evaluation of candidate solutions on test-cases.
 - Parallel evaluation of multiple candidates.
 - Separate judge ranking candidates for beam search and
   final submission selection.


## Components

### Domain-specific Transformer

Architecture:
 - D4 dihedral symmetry and colour permutation symmetry built-in
 - Axial attention
 - *Context* tower
 - Cross-attention between *Cells* and *Context*
   along either or both directions.
 - Enhancements over ViT Transformer block:
   - Grouped query attention (GQA)
   - SwiGLU

Implementation details:
 - Examples weighted (larger examples provide more training signal).
 - Per step dynamic learning rate to account for different batch weights.
 - Gradient accumulation over dynamic number of minibatches
   to counteract large variations in learning rate.
 - Implemented in `flax.nnx` (JAX).
 - Covariant tensor sizes chosen to align with TPU tiling
 - bfloat16


...

### Domain-specific Encoder

- ViT style backbone based on the above transformer
  (S-size: 12 layers of equivariant co-attention)
- Perceiver stack (S-size: 6 layers of *context* attending to *cells*)
- S-size encoder trained as a MAE on ~15 epochs of the
 full pre-computed 400 x 1000k I/O pairs of [Re-ARC]
  (about 150 TPU v6e core-hours).

**Currently implemented**:
- (Training as a classifier predicting task IDs - not used anymore)
- Online evaluation of the embedding quality using k-NN prediction of task IDs from which the image is taken.
- Offline evaluation of the embedding quality using linear probing, again on the task IDs.
- Focal loss: Prioritise *easy* examples - complex tasks
  are out of reach for this architecture.
- Implemented in `flax.nnx` (JAX).

...
