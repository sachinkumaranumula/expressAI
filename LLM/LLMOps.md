# LLMOps
- DevOps for ML is MLOps and MLOps for LLMs is LLMOps

## So far
### MLOps
- One Model per UseCase
- Flow
  - Data Ingestion -> Data Validation -> Data Transformation -> Model -> Model Analysis -> Serving -> Logging
- Automation
  - Job Management
  - Job Orchestration
- Benefits
  - experiment on foundation models
  - Prompt design and management
  - Supervised tuning
  - Monitoring
  - Evaluate generative output

### LLM System Design
- Broader design of the entire end-to-end application (front-end, back-end, data engineering etc)
- Flow
  - UI -> user input -> pre-processing -> grounding -> prompt -> foundation model -> llm response -> grounding -> post-processing -> final output
- Model Customization
  - Data prep -> Tuning -> Evaluate
- Challenges
  - Chaining multiple LLM together
  - Grounding (include facts)
  - Track History

### LLMOps Pipeline
- Steps
  - Orchestration
    - Data preparation and Versioning
    - Pipeline design (Supervised Tuning)
  - Artifact (Configuration and Workflow)
  - Automation
    - Pipeline execution
    - Deploy LLM
  - Prompting and Predictions
  - Responsible AI
  - Missing
    - Prompt Design and Prompt Management
    - Model Evaluation
    - Model Monitoring
    - Testing

- Pipeline Frameworks
  - Airflow
  - Kubeflow*
    - dsl(component, pipeline), compiler


