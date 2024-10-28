# Fwd-Bwd-Prediction
This is towards a working project on understanding the effect of different training objective on the quality of language generation. The project and this Readme file will be updated regularly with new codes.

So far, we look at the two training objectives:
(1) The standard next token prediction with Cross Entropy Loss 
(2) The backward prediction with Cross Entropy Loss. The geneted text is then reversed. If a prompt is given and the goal is completion, then this is done by using the Bayes rule. 

We train two models both on GPT2 like architecture with ~175 M paramaeters on tinystories datasets, with identical hyperparmeter settings and architecture. One does next token prediction in the forward direction and the other in the backward direction. All the arcitecture details, context-size, and optimizaion details (with AdamW) are similar to the standard [GPT2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). The dataset we use is [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by Eldan and Li. The total number of iterations of the optimizer is set in such that it corrreponds to two epochs with TinyStories.

Before executing the code, make sure the dataset `TinyStoriesTrain.txt` and `TinyStoriesTest.txt` are at the path. The training is completed under 3 hours on a GPU node with four NVIDIA A6000 RTX GPUs.
