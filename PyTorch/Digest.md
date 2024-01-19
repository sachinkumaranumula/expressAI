> ### References
> - [PyTorch according to Learning Platform](https://courses.edx.org/courses/course-v1:LinuxFoundationX+LFS116x+3T2022)

# Ecosystem
PyTorch has a very rich ecosystem of tools and libraries. From high-level libraries (e.g. PyTorch Lightning, fastai, Ignite, Catalyst, Skorch) that handle most of the boilerplate involved in developing and training models to highly-specialized ones (e.g. TorchDrug, ChemicalX) for drug discovery, and supporting libraries to improve model interpretability (e.g. Captum) and ensure data privacy (e.g. PySyft), there are libraries and pre-trained models available for a wide range of topics and applications. The range of fields and applications that can be powered by PyTorch is extensive. Computer Vision (Kornia, Medical Open Network for Artificial Intelligence (MONAI), OpenMMLab, PyTorchVideo, Detectron2, PyTorch3D) machine and vehicular object detection, tracking, identification, and avoidance medical image analysis and diagnosis image recognition, classification, and tagging video classification and detection Natural Language Processing (AllenNLP, NeMo, Transformers, flair) text classification, summarization, generation, and translation virtual assistants sentiment analysis question answering and search engines Graph Neural Networks (TorchDrug, PyTorch Geometric, DGL) molecule fingerprinting drug discovery and protein interface prediction social network analysis Spatio-Temporal Graph Neural Networks (PyTorch Geometric Temporal) route planning and navigation traffic control and management inventory planning logistics optimization Gaussian Processes (GPyTorch) time series modeling and anomaly detection risk management control engineering and robotics Reinforcement Learning (PRFL) industry automation and robotics manipulation dynamic treatment regimes (DTRs) in healthcare real-time bidding strategy games Recommender Systems (TorchRec) Interpretability and Explainability (Captum) Privacy-Preserving Machine Learning (CrypTen, PySyft, Opacus) Federated Learning - collaboratively training a model without the need to centralize the data (PySyft, Flower) And then there’s HuggingFace, which is widely known for its large open source community, its model hub, and its array of Python libraries, especially in the area of natural language processing (NLP), since it started as a chatbot company in 2016. We’ll get back to it later in this course.

# Value Proposition
According to McKinsey’s Quantum AI Black "The executive’s AI playbook", the potential total annual value of AI and analytics across industries range from $9.5 to $15.4 trillion. The study covered more than 400 cases spread over 19 industries and nine business functions, and the potential value was split between "Advanced AI" (deep learning) and "Traditional AI and analytics" (machine learning and statistics).

# In the Cloud
PyTorch is supported by the largest cloud providers: Microsoft Azure, Amazon Web Services (AWS), and Google Cloud Platform (GCP). Microsoft Azure It is possible to use PyTorch on Azure in different ways: Azure Machine Learning, a fully-managed instance for training, deploying, and monitoring models Azure Data Science Virtual Machine, a customized virtual machine built for doing data science and evaluating and experimenting with new tools Azure Functions, a serverless solution for deployment with less infrastructure and costs Amazon Web Services (AWS) It is possible to use PyTorch on AWS in different ways: SageMaker, a fully-managed service for training, deploying, and monitoring models AWS Deep Learning AMIs, a customized preconfigured machine instance built for learning, research, app development, and data analytics AWS Deep Learning Containers, a set of Docker images for training and serving models Google Cloud Platform (GCP) It is possible to use PyTorch on GCP in different ways: Vertex AI, a fully managed platform to take models to production Deep Learning Virtual Machine, a compute engine optimized for performance and fast prototyping and experimentation Deep Learning Containers, a set of Docker images optimized for performance, compatibility, and deployment.

# Frameworks
Even though neural networks have been around for a long time, and some frameworks were already available in 2012, such as Caffe, and Theano, it wasn’t until 2015-16 that Big Tech companies started developing their deep learning frameworks: Google’s TensorFlow, Facebook’s Caffe2 and PyTorch, Microsoft’s CNTK, and Apache MXNet (adopted by Amazon).

# The new first step: Transfer Learning
This would be the first step in the commoditization of computer vision models, since these large models, although difficult and expensive to train, could be easily and cheaply fine-tuned to perform different tasks. Once the model was initially trained to perform a more generic task (like classifying images into 1,000 categories), the model was said to be "pretrained", and its configuration (the weights) was made available to the public. Then, anyone could use their own data to train the model a little bit further to perform a different task in the same domain (images). This technique is called "transfer learning", and we’ll get back to it later in this course.

# Models/Algos
PyTorch makes it fairly trivial to use the gradient descent algorithm at scale (think of thousands or even millions of inputs, the columns in our spreadsheet, and millions or billions of data points, the rows) without needing any prior knowledge about calculus or derivatives. Everything is handled behind the scenes by PyTorch’s autograd module, thus allowing the developer or researcher to easily try out increasingly more complex models. One such model is a neural network. Each unit, or neuron, may be thought of as a linear regression too, taking one or more inputs, and producing one single output. This output is then used as an input to yet another unit, or neuron, that is, too, a (different) linear regression. Of course, this is a gross oversimplification: in reality, there are many more bells and whistles, and the outputs undergo transformations before being used as inputs to the next level. These transformations, also called non-linearities, are exactly what makes a neural network both powerful and challenging to train, especially if you stack several layers of neurons on top of each other, turning a neural network into a deep neural network. 