---
title: ExploreJEPA, an Energy based JEPA architecture model
date: 2025-04-27 22:06:35
description: This project designed a JEPA architecture model with CNN and MLPs as encoder and predictor. The main task is to predict the trajetory of an object in an environment with wall and door. Check the code at https://github.com/shawnyin128/ExploerJEPA_an_environment_probing_model.
label: AI
image: '/images/projects/jepa/jepa_cover.png'
page_cover: 
---

# Task
In this project, we aim to build a JEPA-based model to predict the movement of an object in an environment that contains walls and doors. The model needs to learn and reason about the object’s trajectory, as the object can only pass through doors and will be blocked by walls.

# Why JEPA
Joint Embedding Predictive Architecture (JEPA) is a self-supervised learning approach proposed by Yann LeCun in 2022. In his paper A Path Towards Autonomous Machine Intelligence, LeCun suggests that JEPA can facilitate a model’s ability to reason and plan. <br>

We chose JEPA for this project because — aside from the fact that it is required — during inference, the only available information is the initial state and a series of actions. Thus, the model must have strong planning capabilities to predict future states based solely on this limited input, which JEPA is particularly well-suited to handle.

# Data and Data Augmentation
The training dataset contains 2.5 million data points, and each sample is divided into two parts:
- Observation Tensor: Shape [B, T, 2, 65, 65]. <br>
This means there are B trajectories, each trajectory has T time steps, and for each step, there are two 65×65 images representing the object and the environment.
    - Channel 0: Object image ([1, 65, 65])
    - Channel 1: Environment image ([1, 65, 65])
- Action Tensor: Shape [T - 1, 2]. <br>
After the initial state, each following step is associated with an action represented by a 2-dimensional vector.

![Training sample](/images/projects/jepa/data.png)
*Training sample*

Although data augmentation is allowed, we chose not to apply it. <br>

For this particular task, we believe that data augmentation would either have no benefit or could even degrade model performance. Here’s a discussion of some commonly considered techniques: <br>

- Random Crop: <br>
We believe random cropping would be harmful for two reasons:
    - Given the sparsity of the training images, random cropping could frequently produce empty canvases, heavily skewing the training distribution.
    - Random cropping introduces counter-intuitive supervision. Suppose you crop the initial state and obtain an empty image, but the ground truth for the next step involves walls and an object. Then the model would incorrectly learn that an action can spontaneously generate structures from nothing, which violates physical intuition.

- Random Rotation or Translation: <br>
These could also harm model learning, for a similar reason. <br>
Suppose you rotate or translate an image, but keep the original action vector unchanged. In this case, the same action would lead to different visual outcomes under different spatial frames, confusing the model and disrupting the learning of consistent motion dynamics.

- Color Jitter: <br>
Color jitter would neither hurt nor help significantly. <br>
Since the images are grayscale, changing brightness or contrast would have minimal impact, and there is little color information for the model to learn from anyway.

- Noise: <br>
Adding noise is a tricky choice.
If the injected noise is very small — so that the object’s and walls’ visual integrity remains intact — it could potentially help regularize the model. <br>
However, if the noise is strong enough to produce many random grey points, it would confuse the model, possibly leading it to make predictions based on noise artifacts rather than the actual object or environment layout.

- Gaussian Blur: <br>
Similar to color jitter, Gaussian blur is unlikely to cause harm, but it also probably offers little benefit given the simplicity and sparsity of the task.

# Model Structure
The JEPA architecture used in this task follows an auto-regressive prediction framework, as shown in the diagram below.

![JEPA Architecture](/images/projects/jepa/jepa_structure.png)
*JEPA Architecture*
It involves three neural network components:
- A predictor network Pred_ϕ
- Two encoders: Enc_θ for the initial state, and Enc_ψ for target observations.

In our implementation, Enc_θ and Enc_ψ are instantiated as the same network with shared weights, for simplicity — although in general, they can be distinct.

The model operates as follows:
- The initial observation o₀ is passed through encoder Enc_θ to produce the initial latent state s₀.
- Then, using the action u₀ and s₀, the predictor Pred_ϕ produces the predicted next latent state s₁.
- This predicted state s₁ is then used along with action u₁ to predict s₂, and so on — in an auto-regressive manner over time.

In parallel, for each time step:
- The ground truth observation o₁, o₂, ... is encoded by Enc_ψ to generate the target representations s′₁, s′₂, ....

The training objective is to minimize the distance between the predicted latent states sₜ and the encoded ground truth representations s′ₜ at each step, often using a contrastive or L2 loss.

This design allows the model to learn to plan forward purely based on the initial state and a sequence of actions — without relying on access to visual inputs during inference.

## Encoder
The encoder in this task is aimed to extract features from the image. As a result, when design the encoder structure, we need to choose a model that can efficiently work with image. Basically, there are two options, Vision Transformer (ViT) and Convolutional Network (CNN). In our work, we choose CNN. 

There are two reasons we did not use ViT. 
- First, the training set only contains 2.5M data points. According to the ViT paper, they found that ViT can perform better only on large scale of data. On ImageNet which has 1.3M data, its performance is worse than CNN. It starts to outperform CNN under ImageNet-21k which has 14M data. As a result, given 2.5M training data, we don't think ViT can perform better than CNN since CNN has better idea about inductive bias. 
- Second, modern CNN is not worse than ViT even under large training size. In the paper "A ConvNet for the 2020s", they introduce the ConvNext network which outperform ViT both on ImageNet-1k and ImageNet-22k. Our Encoder is basically based on the ConvNext.

## Predictor

# Energy Function

# Train and Result

# Reference
[1] LeCun, Yann. "A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27." Open Review 62.1 (2022): 1-62. [OpenReview](https://openreview.net/pdf?id=BZ5a1r-kVsf) <br>
[2] Project Requirement Repo. [Github](https://github.com/alexnwang/DL25SP-Final-Project?tab=readme-ov-file) <br>
[3] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020). [arXiv:2010.11929](https://arxiv.org/pdf/2010.11929) <br>
[4] Liu, Zhuang, et al. "A convnet for the 2020s." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022. [arXiv:2201.03545](https://arxiv.org/pdf/2201.03545)

