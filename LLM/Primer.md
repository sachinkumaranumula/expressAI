# LLM

- _LM_ : Predict words by looking at word probabilities (probability distribution over entire vocab of tokens and picking the right answer)
- _LLM_ : Just LM's based on transformer architecture with million/billions of parameters

## Building Blocks

- _token_ -> vocabulary (index for tokens) (also the training dataset)
- sequence -> _tokens_...

## Tokenization

character, word, sentence, _sub-word_ (optimal)

## Word Embeddings

- Window(words to left, words to right)
- Algos = _word2vec_: word -> token -> {embedding function} -> vector [large dimensions]
  > These vectors work very well to encapsulate meaning of every token

# Start using LLMs?

- Given a business problem:

  - What NLP task does it map to ?
    - What models work for that task ?

- Tradeoffs
  - Metrics to optimize
    - Cost of queries and training
    - Time for development
    - ROI of LLM-powered product
    - Accuracy of model
    - Query Latency
  - Tips for optimizing
    - Go from simple to complex (existing models -> prompt engineering -> fine-tuning)
    - Scope out costs
    - Reduce cost by tweaking models, queries and configuration
    - Get human feedback
    - Don't over optimize

# Model Gardens

## Hugging Face

- Models
- Datasets
- Spaces (demo)
- Libraries (work w/ datasets, transformers, pipelines, tokenizers, eval)(PyTorch, TensorFlow, JAX)
  - Pipeline (prompt -> tokenizer<encoder> -> model -> tokenizer<decoder> -> output)

### Selecting a model

- Filter
  - task, license, language, model size
  - sort by popularity
  - check release history
- Pick variant of model
  - base model, fine tuned
- Search
  - examples, datasets
- Common Model Families
  - Pythia -> Dolly(databricks), GPT(OpenAI), OPT(Meta), Flan(Google)

### Finetuning a model

- Factors
  - Accuracy (favors larger models)
  - Speed (favors smaller models)
  - Task-specific (favors more narrowly fine-tuned models)
- Approaches (data, performance, cost, size-effect, quality, privacy, lock-in)
  - Few-shot learning
  - Instruction-following LLMs
  - LLMs-as-a-service
  - DIY
- Evaluation
  - Validation Loss
  - Perplexity (confidence)
  - Alignment (Helpful, Honest, Harmless)
  - Task-specific metrics
    - BLEU
    - ROUGE
    - SQuAD

# Common NLP Task

- Summarization
- Sentiment Analysis
- Translation
- Zero-shot Classification (pre-trained)
- Few-shot learning (learn by example) (more of a technique than task)

# Prompts

- Inputs/Queries to LLM to elicit responses
- Engineering
  - model specific
  - good prompt (instruction, context, input, output format)
  - keywords (classify, summarize, extract...)
  - test on examples
- Help
  - Ask model
    - Not to hallucinate
    - Not to assume or probe for sensitive info
    - Not to rush and explain chain of thought
- Format
  - Delimeters between instruction and context
    - "###"
    - "```"
    - "[]/{}"
    - "---"
  - Ask to return structured output
  - Provide a correct example

# Keeping LLMs relevant

- Through model training and fine-tuning
  - Passing Context
    - Analogy: Take exame with open notes
  - Fine-Tuning
    - Analogy: Studying for exam 4 weeks away

# Vector Databases (knowledge stores)

- Embedding Vectors (data -> vectors and onto 2D plane to establish relations between them)
- Usecases
  - Similarity Search
  - Recommendation Engine
  - Security Threats
- Workflow (Search & Retreival Augmentation Generation)
  - Vector Database: knowledge -> embeddings -> vector database
  - Usage: User Prompt -> embeddings -> vector database lookup index -> Context
  - LLM: Use Context against model to generate
- Vector Search
  - K-nearest Neighbors (KNN)
  - Approximate Nearest Neighbors (ANN)
    - trade accuracy for speed
- Similarity between vectors
  - Distance Metrics (higher the metric, less similar)
    - L1(Manhattan)
    - L2(Euclidean) (Popular)
  - Similarity Metrics (higher the metric, more similar) (popular)
    - Cosine of angle between vector
- Compressing with PQ
  - Quantization
    - represent vectors to a smaller set of vectors
  - Process: Big vector -> split into sub vectors -> quantized and assigned to centroid (repeat)
- Search
  - Algos (Vector Indexing)
    - FAISS (cells and centroid)
    - HNSW (graphy layers local minimum)
- Filtering
  - Post Query (ANN -> filter)
  - In Query (scalars in vector, memory intensive)
  - Pre Query (brute force vector space)
- Vector Stores
  - database
    - organize embeddings into indices
  - libraries
    - small data (no db props like crud, no disk store etc...)
  - plugins
    - databases providing ANN
- Need?
  - Pro (Speed, Scalability, db kind)
  - Con (Cost, one more in the stack)

# Multi-stage Reasoning

## Why?

- LLM's can do classical NLP tasks but most workflows have set of tasks where just one task can be an LLM task
- e.g. Summarize and Sentiment
  - Prompt: Summarize the following article, paying close attention to emotive phrases: {article}

## How?

- LLM Chain
  - Chain tasks to form a workflow chain
  - e.g. LLMMath in LangChain
    - output from llm with code -> evaluate the code -> record the output

## Composition

- Agent
  - LLM based systems execute _ReasonAction_ loop i.e. giving LLMs the ability to delegate tasks to specified tools
  - Building Agent
    - Building Blocks
      - Task: Do this thing
        - Tools: Use these tools to complete this task (LLM will select and execute to perform steps to achieve the task )
        - LLM: The Brain (reasoning or decision making entity)

# Risks and Limitations

- Data
  - Big Data != Good Data
  - Discrimination, Exclusion, Toxicity
- Unintentional Misuse
  - Information Hazard
  - Malicious uses
- Society
  - jobs
  - environment

## Hallucinations (_Tricky & Under Research_)

- Types
  - Intrinsic
    - Output contradicts Source
  - Extrinsic
    - Cannot verify Output from Source
- Causes
  - Data
    - data without factual verification
    - duplicates
  - Model
    - imperfect encoder learning
    - erroneous decoding
    - exposure bias
    - parametric knowledge bias
- Evaluation
  - Statiscial Metrics
    - BLEU, ROUGE, METEOR, BVSS
  - Model-based Metrics
    - QA-based, LM-based, Information Extraction, Faithfulness
- Mitigation
  - Data
    - Build faithful dataset
  - Model
    - Research & Experiment (Reinforcement, Multipass learning)

## LLMOps

- MLOps = DevOps + DataOps + ModelOps
- LLMOps = MLOps w/ LLMs (change in objects, workflow, ops)
- MLFlow = ML lifecycle (models, tracking, registry) (different libraries -> _MLFlow_ -> deployment options)

> _References_
>
> - Natural Language Processing
>   - [Stanford Online Course on NLP](https://online.stanford.edu/courses/xcs224n-natural-language-processing-deep-learning)
>   - [Hugging Face NLP course](https://huggingface.co/learn/nlp-course/chapter1/1)
> - Language Modeling
>   - [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
>   - [Bag of Words](https://www.kaggle.com/code/vipulgandhi/bag-of-words-model-for-beginners)
>   - [LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
>   - [Language Modeling](https://web.stanford.edu/~jurafsky/slp3/)
> - Word Embeddings
>   - [Word2vec](https://www.tensorflow.org/tutorials/text/word2vec)
>   - [Tensorflow Page on Embeddings](https://www.tensorflow.org/text/guide/word_embeddings)
> - Tokenization
>   - [Byte-Pair Encoding](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt)
>   - [SentencePiece](https://github.com/google/sentencepiece)
>   - [WordPiece](https://ai.googleblog.com/2021/12/a-fast-wordpiece-tokenization-system.html)
