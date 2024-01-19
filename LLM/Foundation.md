# Transformer Architecture

- What CNN did to Vision Models, Transformers did to Language Models
- "Attention is all you need" - how words interrelate in a sequence

## Transformer Block

- Preparation
  - Take input sequence of tokens and transform into word vectors (series of word vectors)
  - Add positional encoding (gives relative position of token with others)
- Enrichment (transformer block)
  - Add contextual information using attention mechanism and non linear transformation by neural network
- Prediction
  - Add residual connection and normalizing vectors in sequence and those vectors are used in linear and softmax combination to predict next token or classify the sequence

### Attention

- Measure importance and relevance of each word relative to each other
- Allow for building-up of enriched vectors with more context and logic
- Built out of three vector families
  - Calculate Attention
    - Step 1: Input Vector(word embedding + positional encoding) -> Three new vectors
      - Query Vector (current token)
      - Key Vectors (all other tokens in sequence)
      - Value Vector (all tokens in sequence)
    - Step 2: Attention Weights (parameters)
      - Scaled dot product query vector and key vectors (scale: 0-1)
    - Step 3: Output Vector
      - Each value vector multiplied by attention weights
  - Analogy
    - Filing cabinet and lookup system
      - for a query(query vector) and we are looking through the files how well each of the other files(key vector) have the information that we need(value vector)
      - once we figured out how much each key vector should give to the query vector i.e. attention weights we then combine altogether to get a full picture(output vector) i.e. how much attention to pay to other tokens in the sequence w.r.t to current token

### Position-wise Feed-Forward Networks

- tokens given to transformer are turned into word embeddings of n-dimension long, n neuron wide neural network
- each token in passed to neural network one by one which is fed forward into next transformer block

### Residual Connections

- un-interrupted path to next layer with same structure to mitigate vanishing gradient problem in deep neural networks

### Layer Normalization

- stabilizes network training, ensuring consistent input distribution crucial in tasks with varying sequence lenghts

## Input and Output of Transformers

- Input
  - From original sequence to new sequence with each element a dense vector
- Output
  - From a sequence of context-aware vector
  - Vocabulary and linear neural network that select next token using softmax function based on sequence of vectors or classify it using classification scheme

# Variants

- Original Transformer
  - Encoder Decoder
    - two sets of blocks
      - encoder blocks
      - decoder blocks
      - cross attention
  - Usage
    - Translation
    - Conversion
- Encoder Only
  - BERT (Bi-directional Encoding Representation)
    - Segment Embedding
    - Usage
      - Q&A
      - Named Entity Recognition
- Decoder Only
  - GPT (Generative Pre-Trained)
    - Predict next word
    - Usage
      - Generative

# Important Variables

- Input
  - V: Vocab size
  - D: Embedding/Model size (word embedding dim)
  - L: Sequence/Context Length (number of tokens in single pass to model)
- Internal
  - H: Number of Attention Heads (multi attention hidden layers - splitting k,v vectors into separate with each head focuses on different parts of speech)
  - I: Intermediate Size (feed forward network size)
  - N: Number of Layers (number of transformer blocks/layers)
- Training
  - B: Batch Size (training examples passed forward/backward)
  - T: Tokens Trained (total number of tokens model sees during training)

# Foundation Models

## Training

- Model
  - Task specific
  - Arch
    - encoder(BERT)
    - encoder-decoder(T5)
    - decoder(GPT)
  - Sizes: dimensions, number of T-blocks
- Available Data
  - The Pile
  - BYO (only reason for your own training)
- Available Compute
  - months on hundreds of GPU

## Alignment

- Accuracy
- Bias/Toxicity
- Lack of task specific focus

## How was it done?

- Journey of GPT
  - GPT (family of models) from Encoder-Decoder paper focus on only Decoder phase
    - 1
      - N: 12
      - D: 512
      - Dataset: BooksCorps
    - 2
      - N: 48
      - D: 1024
      - Dataset: WebText
    - 3
      - N: 96
      - D: 2048
      - Dataset: WebText2
    - 4
      - assumed (not released by OpenAI)
  - Why so many Layers? (N)
    - Early Attention
      - Word Order
      - Parts of Speech
    - Middle Attention
      - Meaning
      - Relationship
      - Semantic info
    - Late Attention
      - High-level Abstractions
      - Sentiment
      - Discourse Structure
  - Why so many Parameters?
    - N: Layers increasing
    - D: Dimensionality increasing
    - H: Number of attention heads

# Fine Tuning

- Subset of Transfer Learning (training even more or training on different data)
- Approaches
  - PreTrained: Training data -> Foundation Model -> Predictions (GPT-4, T5)
  - Feature Extraction: Training Data -> Foundation Model -> Output Embeddings -> Model -> Predictions (BERT embeddings)
  - Fine Tuning: Training Data -> Foundation Model -> Fine Tune (top layers, all layers, add layers) -> Predictions
- Why?
  - Improve performance
  - Ensure Regulatory Compliance
- What?
  - Update foundation model weights
- How?
  - _Full fine tuning_ (expensive beast memory rich GPU)
  - _X-shot learning_ (not updating model weights)
    - prompt engineering
    - in context learning
    - pros
      - no need for labeled training data
      - same model for every task (simplified model serving)
    - cons
      - manual and labor intensive
      - highly specific to models
      - context length limitation
      - performance issues
    - e.g: FLAN, Dolly
  - _Parameter efficient fine-tuning_ (PEFT)
    - Additive
      - add new tunable layers
      - soft prompts
        - task specific virtual tokens (same length input embeddings) and are not part of vocabulary
        - back prop helps find best representation of prompt
    - Re-parameterization
      - decomposing weight matrices into lower rank matrices
      - LoRA
        - Matrix Rank: unique row and column
        - Weight Matrix Decomposition: Actual rank of attention weight matrices is low
        - E.g.: total params (100,100) represented by rank (100,2) and (2,100)
      - ability to share and reuse foundation models (merged weights)
      - not straightforward when mixed task batch

# Deployment and Hardware Considerations

- Improving Language Efficiency
  - Alibi
    - adds a linear basis function to the attention calculation (read paper)
  - Flash Attention
    - Attention scores -> matrices -> SRAM of GPU (limitation)
    - flash attention deals with just the needed indices
  - Many Queries, fewer keys
    - Multi-headed (inference: slow, accurate)
    - Multi Queries (inference: fast, inaccurate)
    - Grouped Queries (inference: fast, accurate)
- Improved Model Footprint
  - Floating point precision: Google Brain FP16, FP32, BF16
  - Quantization factor: buckets of ranges (approximate fp values in quantized forms)
  - QLoRA: Applying Quantization to tuning
- Multi-LLM Inferencing
  - Mixture of Experts (MoE)
    - Input is sent to router, multiple NNs are trained
  - Switch Transformer
    - Application of MoE
  - LLM Cascades and FrugalGPT
    - send prompts, analyze confidence score, move from smaller to larger models (budget oriented)
- Best Practices for Training from Scratch
  - ALiBi
  - Flash Attention
  - Grouped-Query Attention
  - Mixture-of-Expers
  - LoRA/QLoRA
  - FrugalGPT

# Multi-modal LLMs

- E.g. VideoLLaMA, CLIP, MiniGPT4, Stable Diffusion
- Transformers beyond text
  - cross-attention can bridge between modalities
  - _Vision_
    - image to rgb hex codes
    - embedding of word are 2d tensors while images are 3d tensors (height, width, channels)
    - Models
      - Vision Transformer
        - ViT outperformed CNN by 4x
        - image to patch embeddings
        - turns an image into sequence of patches
        - patch sequence with positional embedding goes through transformer
      - Zero shot learning
        - CLIP
      - Few shot learning
        - Flamingo
  - _Audio_
    - embedding vectors for each t-second audio frame
    - Data2Vec, Whisper

# Contending Architectures (beyond attention)

- RLHF (Re-inforcement learning w/ human feedback)
  - human feedback trains reward model, KL loss ensures minimal divergence from original model and Proximal Policy Optimization(PPO) updates the LLM
- Hyena Hierarchy
  - CNN matching performance of ViT
- Retentive Networks
  - a new attention variant, it is able to achieve higher computational efficiency without affecting performance
