> ### References
>
> - [McKinsey](https://www.mckinsey.com/business-functions/quantumblack/our-insights/breaking-away-the-secrets-to-scaling-analytics)
> - [Gartner](https://www.gartner.com/smarterwithgartner/how-to-improve-your-data-quality)

# Data Strategy

- a clear data ontology and a corresponding master data model
- governance plans to assign responsibility for the quality and maintenance of datasets
- understanding and planning the necessary technical requirements regarding data collection and availability

# Data Governance

- Starting at the link between business processes, KPIs, and data assets, the first set of steps advocates for a strong foundation and focused approach with a clearly defined scope, and a standardized definition of "good enough" data quality across the organization.
- Going into the next level, the second set of steps advocates for ongoing data profiling in order to identify data quality issues combined with dashboards for monitoring critical data assets.
- The third level outlines the steps to assign accountability for data quality, linking data quality initiatives to business outcomes, establishing responsibilities and operating procedures, and building collaboration between business units, IT, and the Chief Data Officer.
- Finally, in the last level, the stage is set to embed data quality improvement into the organizational culture through the periodical communication of impactful best practices and reaped benefits.

# Data Quality

## Problem Statement

Ensuring data quality is a tall order, though, since data can be flawed in a variety of ways:

- missing: the data is simply absent
  - empty fields in a form
  - failure in data collection
  - failure in data recording
- incorrect format: the data is there, but it’s in the incorrect format, and the actual data is likely missing
  - absent validation/sanitization of inputs, e.g. the "DOB" reads "Joe" instead of an actual date
  - conversion error due to localization issues (decimal point, date format)
- erroneous: the data looks like a valid entry, but it’s likely in the wrong range, scale
  - fat-finger errors
  - error in measurement due to a faulty sensor
  - wrong unit of measurement used in the input, e.g. meters instead of feet
- inaccurate: the data is not flawed per se, but it’s not suitable for its intended use
  - data is stale, that is, it is not representative of the problem it is being used to solve anymore
  - data is a poor proxy: in some cases, the true data isn’t available or it’s not even possible to observe or collect it, so a proxy -(the "next-best-thing") is used in its place; but there’s no guarantee a good proxy exists either
  - data may be of subjective nature (e.g. ratings) and quantifying it numerically is an approximation only

## C.A.R.E

### Connected

The data is connected if it does not have any "gaps" in it, that is, there are no values missing or values in an incorrect format. Disconnected data is harder to handle because most algorithms cannot handle them, decision-tree-based algorithms being the most notable exception. This means records containing missing data must be either discarded or these values must be imputed. The former leads to loss of information while the latter leads to assumptions and biases being introduced in the data, as discussed in the previous section.

### Accurate

The data is accurate if it is a proper measure of what it represents, that is, if it is timely, in the expected unit, there are little or no measurement errors, and actually represents the underlying phenomenon. Inaccurate data may break an application in a silent way since values wildly outside of the expected range may skew the model during development, thus impacting performance, or it may produce absurd predictions if used as input to a model in production. The former leads to a sharp drop in the value created by the model while the latter may lead to a loss of trust in the model’s ability to deliver sensible predictions.

### Relevant

The data is relevant if it is actually related to the task at hand, that is, if the data causes, or at least correlates, to the phenomenon it is being used to describe. For example, if we’re trying to estimate the level of sugar in one’s blood, it makes sense to look for some data related to one’s food intake (e.g. size and number of donuts in the last 24 hours), but the actual price of the donuts is irrelevant in this case. Even though irrelevant data may be weeded out using techniques like feature selection, the indiscriminate use of each and every piece of data available ("let’s try everything and see what sticks") is discouraged. In some cases, two completely unrelated sources of data may be strongly correlated. These are spurious correlations, and although they may look real, they are nothing but a statistical fluke. If you look hard enough, you will eventually find two highly correlated streams of data, like the example below: Divorce rate in Maine vs Per capita consumption of margarine Divorce rate in Maine vs Per capita consumption of margarine (retrieved from Spurious Correlations)

Could we train a model to predict the divorce rate in Maine using the per capita consumption of margarine as an input? Of course. Would the model be a good representation of an actual phenomenon (the divorce rate in Maine)? Obviously not.

### Enough to Work with

"Enough" is a relative concept when it comes to data. The more complex the model (and deep learning models are the most complex ones), the higher the bar for data to be considered "enough" data.

On the one hand, the more data, the better (as long it’s high-quality data). On the other hand, sometimes data acquisition and collection may be difficult or expensive. If that’s the case, it may be more cost-effective to either choose a less-complex model, or leverage techniques such as data augmentation, semi-supervised learning, or transfer learning.

# Data Structure

### Structured

Garden-variety tabular data. It is well-organized in rows and columns, and it can easily be stored in a relational database (e.g., Oracle, SQL Server, PostgreSQL) to be queried using SQL (structured query language). Structured data is estimated to account for 20% of all the data by 2025

### Unstructured

Consists of images, audio, video, and text in their raw, native, formats. Emails, PDFs, Word documents, PowerPoint presentations, photos, screenshots, and YouTube videos are all examples of unstructured data. This type of data makes up the bulk of the humongous amount of data generated and consumed every day by Internet users. Unstructured data is estimated to account for 80% of all the data by 2025. Deep learning models shine on this type of data, as we’ve seen with computer vision and natural language models in previous sections, but only if the data is accompanied by the corresponding label(s), thus making it semi-structured data.

Semi-structured data bridges the gap between unstructured and structured data by adding metadata (e.g. tags, labels) to identify and describe the characteristics of the underlying unstructured data. Excel spreadsheets, JSON, CSV, and XML files are examples of semi-structured data since they exhibit an internal structure (e.g. column headers in a CSV file) but that structure is not a predefined one. The internal structure can be used to define a schema such that the data can be imported to a relational database, thus turning it into structured data at the potential cost of creating lots of missing data in the process.

# Data Labeling

Labels, or annotations, may indicate a variety of attributes of the data they are describing: content, location, position, role, etc. These labels can be used to catalog and search the data, but their main purpose when it comes to machine and deep learning models is to be the "target" of that piece of data.

## Annotating Images

The first, and most common type, of labeling images is assigning a category (e.g. "swan") to images containing a single object or entity. The label can then be used as a target for an image classification task.
If there are many objects in the same image, each object of interest must have its location in the image annotated (usually drawing a box around it – a _"bounding box"_) and the content of each box is assigned to the category of the object it contains (e.g. "bottle", "chair", "computer mouse"). The boxes and labels can be used as a target for an object detection task.
It is also possible to focus on separating objects in an image from one another and the background using its natural borders instead of drawing a rectangular box around them. Each separate region of interest can be used as a target for a _semantic segmentation_ task.
Finally, you can annotate an image of a person to highlight the pose they’re in. The annotator draws a "stick figure" over the person of interest in the image highlighting several key points in the human body. The coordinates of these keypoints, together with the actual pose (e.g. sitting, standing, etc.) can be used as a target for a _pose estimation_ task.

There are several approaches to the labeling/annotating process:

- manual annotation, where the annotator has to draw regions and assign labels to each image
- programmatic or semi-automated annotation, where tools are used to automate the selection of regions/pixels which can be corrected by the annotator if needed
- synthetic labeling, where images corresponding to the required labels are synthetically generated (e.g. generative models)

## Annotating Text

One of the most common types of labeling when it comes to text is categorizing the nature of the whole document (e.g. "invoice", "report") or its subject area (e.g. "legal", "biology"). The definition of "document" is very broad: it does not necessarily mean a file as a whole, and it may refer to individual chapters, sections, paragraphs, or sentences. This type of annotation is used for _document classification_ tasks.

Another type of document classification is called _sentiment analysis_, where each document must be labeled according to the emotion, opinion, or sentiment inherent in its body. It is commonly used to assess customer reviews or social posts as positive, neutral, or negative.

If we get past sentences and down to the level of words, annotations can be used for a variety of purposes:

- named entity recognition (NER): entities with proper names (e.g. persons, cities, countries, companies)
- part-of-speech (POS) tagging: syntactic functions (e.g. verb, noun, adjective)
- keyphrase tagging: the location of keywords or phrases of interest
- linguistic annotation: grammatical, semantic, or phonetic elements

These annotations are used to train models used across many applications, like virtual assistants, chatbots, and search engines.

# Learning

Data labeling is an integral part of application development since it is required to train most models: we need to provide them with both input data and expected output so that they can learn the underlying association rules by themselves. The technical term for this class of problems, where labels are required, is _supervised learning_

_Unsupervised learning_, as you probably guessed, does not require labels. This class of problems is commonly addressed by using clustering algorithms to group data points together (in clusters) according to some similarity metric. They uncover hidden patterns in data without any need for human intervention (e.g. labeling), that is, without any supervision. Typical examples of applications are anomaly/fraud detection and recommender systems.

Then, there is _semi-supervised learning_, where a small portion of the data is labeled, but the rest is not. The general idea behind it is to leverage the existing labels to train an initial model and then use this model to, provisionally, label (part of) the remaining data. The newly labeled data is then used to further train the model, and so on, and so forth.

_Transfer Learning_ - Transfer learning is one of the main drivers of the commoditization of deep learning models. It leverages the power of pre-trained models to deliver better performance from the get-go (when compared to training from scratch) and higher performance potential.

# Feature Engineering

> "Domain Expertise Meets Insights from Data"

The process that transforms raw data into features, that is, into numerical attributes or characteristics with certain properties, is called feature engineering. This is a hard requirement for most algorithms since they cannot handle well (or at all) qualitative information, missing values, or values on wildly different scales.

Some approaches: _One-hot encoding_, _normalization_, _Imputation_, _synthetic features_

# Model

Model won’t ever accurately and completely describe all nuances and complexities of reality, otherwise, it wouldn’t be a model – the model’s purpose is to simplify. A model is an abstraction, an idealized form, of an underlying process that takes place in reality. On the one hand, it is exactly the simplicity of a model (when compared to reality) that makes it valuable. On the other hand, reality is simplified through assumptions, and a model is only as good as the validity of the assumptions used to build it.

You may be tempted to believe that a more-complex model will always outperform a less-complex one. The reality, though, isn’t as straightforward as this. The more complex a model is, the more data it requires, and we’ve already discussed the costs and challenges of data collection, acquisition, and labeling. Moreover, complex models may also fail to deliver on other requirements such as latency, explainability, and interpretability.

## Type

Deep learning models are _black-box_ models, that is, they are highly complex and performant, but it’s virtually impossible to pinpoint exactly what’s happening under the hood: they are empirical, data-driven, models.

Traditional statistical models, on the other hand, are exactly the opposite: they are analytical, _white-box_, models. They start from first principles and they are focused on accurately and properly describing a problem while taking causal effects into consideration. The main concern is to find the "correct" model, as opposed to the most performant one. These models are likely simpler and their inner mechanisms can be thoroughly understood but it usually comes at the cost of performance (when compared to a black-box model).

## Quality

**Train Validation Test Split** - To properly evaluate the performance of a model, data scientists hold out part of the available data and use it for evaluation only: that’s called a _validation set_. The model never sees this data during training. Once the model is fully trained, the validation set is used to assess the model’s performance on previously unseen data. The results of this assessment are then used to inform other technical choices (hyper-parameters) for retraining the model. Unfortunately, this back-and-forth makes the validation set a poor choice to assess the expected model performance after deployment. Enter the _test set_, yet another part of the available data that should be held out with the single purpose of emulating the future.

## Interpretability and Explainability

> _Ref_
>
> - https://docs.aws.amazon.com/whitepapers/latest/model-explainability-aws-ai-ml/interpretability-versus-explainability.html
> - https://christophm.github.io/interpretable-ml-book/agnostic.html
> - https://cloud.google.com/blog/products/ai-machine-learning/explaining-model-predictions-on-image-data

Interpretability and explainability can be easily misunderstood as synonyms but, in the context of modeling, they are fundamentally different. Unfortunately, these terms are often used loosely, only adding to the confusion around them.

Interpretability (or the lack of it) is closely related to what we discussed in the previous section: white- and black-box models. It answers the questions of "why" and "how" a model is producing a given output. Interpretability means that the cause and effect can be determined (the "why"), and it is possible to understand how exactly the model went from one to the other (the "how"). White-box models, since they’re built from first principles, are more interpretable than black-box models.

Explainability, on the other hand, only answers "how" a model is producing a given output. Once again, white-box models are more easily explainable, but even black-box models can be explained if you use some clever techniques to peek inside them. It is important to notice that, even if you can explain "how" a black-box model arrives at a given conclusion (e.g., subject A is likely to default), there’s rarely (if ever) an indication of "why" it did it.

## Vision

There are many different pre-trained models for computer vision. The most popular ones, such as VGG, Inception, and ResNet, were developed for the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) between 2010 and 2017. The challenge had 1.2 million images split into 1,000 different categories (labels), and the goal was to train a model able to correctly classify images into these categories. These models, and their subsequent variations and improvements, are at the heart of transfer learning for computer vision. They are widely used as pre-trained base models for visual applications in general.

## NLP

Word embeddings are numerical representations of words. Given a fixed-size vocabulary (e.g., 100,000 English words), a word2vec model could build a numerical representation (a sequence of 50, 100, or 300 numbers, the vector) of a word by training it to predict which word fill in a blank based on the words surrounding the blank position.

Then, you could use those embeddings – large lookup tables – to convert sequences of words into numerical values, and then use these values to train a model to accomplish a given task. It wasn’t true transfer learning since the model was still being trained from scratch, although using a conversion (from words to numbers) learned by another model.

It was only in 2018 that transfer learning started to become a reality for natural language processing tasks. Models like ELMo (an acronym for Embeddings from Language Models) and the original Transformer, and later on BERT (Bidirectional Encoder Representations from Transformers) and the many versions of GPT (Generative Pre-trained Transformer), completely transformed the landscape. The Transformer architecture, which we’ve briefly mentioned before, was a major breakthrough, and it has been successfully used not only in natural language processing tasks but also in computer vision ones. Nowadays, there’s a large number of pre-trained language models that can be used as a base for developing applications.

## PreTrained Model Hub

The HuggingFace Hub is your one-stop shop for pre-trained PyTorch models for both computer vision and natural language processing tasks and more. It is a platform with over 60,000 models, all open source and publicly available. It also supports dozens of libraries, many of which are an important part of the PyTorch ecosystem, such as HuggingFace’s own Transformers, AllenNLP, and flair.

# Concerns

## Assessing Feasibility

Unlike traditional software, the development of AI applications resembles, in many ways, a research project. It is of the utmost importance to clearly specify a success metric and a minimum "good enough" performance threshold for deployment

There are two sides to assessing the feasibility of an AI application. First, the technology must be available, that is, there must be an algorithm or model that has the potential to successfully address the task at hand. Second, there must be enough high-quality data available.

## Adversarial Inputs and Attacks

AI security is a hot area of research, and there are libraries built to help develop defenses against adversarial attacks, such as the [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) and Microsoft’s [Counterfit](https://www.microsoft.com/en-us/security/blog/2021/05/03/ai-security-risk-assessment-using-counterfit/). Moreover, [MITRE](https://atlas.mitre.org/), in collaboration with Microsoft, released MITRE ATLASTM (Adversarial Threat Landscape for Artificial-Intelligence Systems), an extensive knowledge base that includes tactics, techniques, and case studies, one of which is Microsoft’s Tay poisoning that we’ve discussed earlier.

## Data Privacy and Federated Learning

The centralization of data in a single location, though, poses a high security risk, should the organization be the target of hackers and have its data leaked. Moreover, as data privacy laws grow stricter, it will get increasingly harder for companies to collect, store, and leverage user data for their own purposes. Federated learning addresses both concerns.
Furthermore, for more sensitive data, the level of access to the local data can be set using differential privacy. The idea behind differential privacy is to only share aggregates (e.g.. summary statistics) of the dataset, thus protecting the privacy of individual records that are part of it. An algorithm is said to be differentially private if you cannot tell whether the information of a particular individual was used or not in the computation.
Finally, the use of homomorphic encryption allows computation to be performed over encrypted data. In other words, the result of an operation over encrypted inputs is the same as the encrypted result of an operation over raw inputs. For example, if numbers were "encrypted" as colors, adding up two colors should produce the color corresponding to the "encrypted" result of the mathematical operation

## Concept Drift

Model development is not software development. A model is never truly finished. Once it is deployed as a real-world application, its performance immediately starts to degrade, the rate of decay depending on the nature of the problem it was trained to address.

"Past performance is no guarantee of future results," as stated in fine print in most, if not all, financial products. On the one hand, the model is built upon the assumption that the relationship between inputs and outputs remains static over time. It couldn’t be any different – the future is unknown. On the other hand, the world keeps on turning, and the underlying conditions and assumptions upon which the model was trained change. This is called concept drift.

## "Wrong" or Anomalous Inputs

Models don’t know how to say "I don’t know" or "I have never seen anything like it". They will always output a prediction, no matter how bad, weird, or different the given input is.

A "wrong" input is an input that clearly does not belong to the same domain the model was trained to handle.

## "Bad" Quality or Shifted Inputs

While wrong" or anomalous inputs belonged to a domain completely different from the domain the model was trained for, "bad" quality inputs belong to a domain only slightly different. To avoid issues related to domain shift (also called distributional shift), the images used during training must be representative of the real-world conditions the end-users will submit to the model for classification. In those cases where collecting (and labeling) more data from the real-world use of an application is possible and feasible, you could use transfer learning to fine-tune and improve the model’s performance.

## Monitoring your Model

Proper monitoring may be time-consuming and, while everything is working fine and as expected, it may even be considered a nuisance, a waste of time and resources, and downright boring. Nothing could be farther from the truth. At the end of the day, your users will act on the predictions produced by your model, and it is your duty to ensure it is performing as expected: if not by the force of regulations, for the sake of your reputation and the trust your users thrust upon your model.

To address these challenges, the industry developed machine learning operations – MLOps – which is the application of DevOps principles to AI-powered applications, taking into account AI-specific requirements and limitations. The goal of MLOps is to streamline the process of deploying, maintaining, and monitoring AI models in production. It allows organizations to more safely test and deploy new models without disrupting operations (that rely on the already deployed model) while, at the same time, reproducing and investigating possible issues that may arise, such as biases or wrong predictions, for example.
