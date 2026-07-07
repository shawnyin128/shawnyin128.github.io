---
layout: post
title: "From SFT to RL: The Two Degrees of Freedom"
date: 2026-07-06 21:00:00 -0400
slug: "Post-Training from First Principles: SFT"
description: "Why SFT is reference-distribution fitting, and how changing token weights and sampling turns it into RL."
tags: [LLM, Post-Training]
math: true
---

Many people, when first trying to understand Supervised Fine-Tuning (SFT), think of it as another training stage after pretraining: pretraining is self-supervised (SSL), while SFT is supervised. There is nothing wrong with this description, but I have always felt that there is a detail worth thinking through: **if SFT is still using cross-entropy loss to predict the next token, then how exactly is it different from SSL?**

This blog is an attempt to answer that question. Instead of starting by saying what SFT is, we will first **take it apart**: how the prompt and response are organized, **why the loss is usually computed only on the response**, and how these “reference answers” enter the training objective in the first place.

## SFT Is Language Modeling with a Mask

SFT is still language modeling. More specifically, it is still **autoregressive next-token prediction**: given the preceding tokens, predict the next token. For a sequence of text $(x_1,\dots,x_T)$, the model factorizes the probability of the whole sequence into a product of conditional probabilities at each step:

$$
p_\theta(x_1,\dots,x_T)
= \prod_{t=1}^{T} p_\theta(x_t\mid x_{<t})
$$

This factorization matters because it tells us what a language model is actually trained to do: **each token is predicted conditioned on its prefix**. The log probability of the full sequence is the sum of these per-token log probabilities.

SFT keeps the same training form, but changes **how the training example is organized**. A typical SFT example is split into a **prompt $x$ and a response $y$**. If we concatenate the two, it is still just a sequence of text. If we trained it exactly like ordinary language modeling, the model would be modeling the full joint distribution:

$$
p_\theta(x,y)
= p_\theta(x)p_\theta(y\mid x)
$$

**Masking out the loss on the prompt** means we no longer ask the model to explain $p(x)$; we only ask it to explain $y$ given $x$. Therefore, SFT learns the **conditional distribution $p(y\mid x)$**, not the full joint distribution $p(x,y)$. This also matches how the model is used in practice: the prompt is provided by the user, and the model needs to learn how to respond after that prompt.

So where does the “supervision” come from? Not from the loss function itself, but from the **source of the target tokens**. Pretraining also has labels, but those labels come from the corpus itself: the next word is already written in the text. In SFT, for the same prompt, what counts as a good answer, and which answer the model should learn from, is specified by a human, a stronger model, or some data generation pipeline. In other words, **the supervision is in the data, not in the mathematical form**. The loss has not changed; what has changed is who decides the answer. This is why pretraining, SFT, and instruction tuning may look like different stages, but at the bottom they are all still next-token prediction. The main differences are the **data source** and the **loss mask** [1].


## SFT Fits the Reference Distribution

SFT is a clean way to **align the training objective with the generation setting**: given a prompt, learn to produce the kind of response we want. But the same formulation also shows its **mathematical ceiling**. Once the training signal comes from a **fixed reference distribution**, SFT can only move the model toward that distribution.

Denote the **reference distribution** behind the training data as $q(y\mid x)$. Here, $q$ may be induced by human annotators, a stronger model, or some data generation pipeline. What SFT does is **minimize the negative log-likelihood (hard-label cross entropy)** on samples drawn from this reference distribution:

$$\mathcal{L}(\theta)=\mathbb{E}_{y\sim q}\big[-\log p_\theta(y\mid x)\big]$$

Expanded autoregressively, **the loss on the response is a sum over tokens**:

$$-\log p_\theta(y\mid x)=-\log\prod_{t}p_\theta(a_t\mid x,a_{<t})=-\sum_{t}\log p_\theta(a_t\mid x,a_{<t})$$

Here, $a_t$ is the $t$-th token in the response, and $a_{<t}$ is the part of the response that has already appeared before it. We will use this **token-level form** when looking at the gradient later. For now, treating $p_\theta(y\mid x)$ as a whole makes the objective of SFT easier to see.

Now apply a standard **KL decomposition** to this objective:

<details class="math-note" markdown="1">
<summary><b>KL Decomposition Derivation</b></summary>

Inside the $\log$, multiply by $1=\dfrac{q}{q}$:

$$-\log p_\theta=-\log\!\Big(p_\theta\cdot\tfrac{q}{q}\Big)=-\log\!\Big(\tfrac{p_\theta}{q}\cdot q\Big)$$

Use $\log(ab)=\log a+\log b$ to split the expression:

$$-\log\!\Big(\tfrac{p_\theta}{q}\cdot q\Big)=-\log\tfrac{p_\theta}{q}-\log q$$

Flip the ratio in the first term:

$$=\log\tfrac{q}{p_\theta}-\log q$$

Substitute this back into the expectation:

$$\mathcal{L}(\theta)=\mathbb{E}_{y\sim q}\Big[\log\tfrac{q}{p_\theta}\Big]+\mathbb{E}_{y\sim q}\big[-\log q\big]$$

The first term is the KL divergence, and the second term is the entropy of $q$:

$$\mathcal{L}(\theta)=\underbrace{\mathbb{E}_{y\sim q}\Big[\log\tfrac{q}{p_\theta}\Big]}_{\mathrm{KL}(q\,\|\,p_\theta)}+\underbrace{\mathbb{E}_{y\sim q}\big[-\log q\big]}_{H(q)}$$

$H(q)$ is independent of $\theta$; $\mathrm{KL}\ge 0$, and it is zero only when $p_\theta=q$.

</details>

$$\mathcal{L}(\theta)=\mathrm{KL}(q\,\|\,p_\theta)+H(q)$$

$H(q)$ is the entropy of the reference distribution itself, so **it does not depend on the model parameters $\theta$**. In other words, **minimizing the SFT loss is equivalent to making $p_\theta$ as close as possible to the reference distribution $q$**. In the ideal case, **the optimum is $p_\theta=q$**.

The meaning is simple: if the training signal comes only from this reference distribution, then **SFT is trying to fit that distribution**. It is **not directly searching for a policy that is “better than the reference distribution”**; it is fitting the distribution of answers that appear under $q$. This also explains the connection between **SFT and knowledge distillation** [2]. Mathematically, both optimize cross-entropy against some reference distribution $q$. The difference is where that reference distribution comes from: if it comes from a teacher model, we usually call it distillation; if it comes from human-written answers or filtered data, we call it SFT. The names are different, but the optimization objectives are very close.

This perspective is also useful when reading experiments. Many papers use a stronger large model to generate data, then SFT a smaller model on that data, and finally report improvements on benchmarks. There may well be methodological contributions in such work, but the first thing to acknowledge is that **this pipeline is essentially learning from a teacher-induced reference distribution**. A natural baseline should be **“directly distill the same teacher.”** If a method cannot beat that baseline, the gain may be coming mostly from the data source, rather than from the new method itself.

## Breaking SFT's Ceiling

In previous section, we saw that SFT is a clean training paradigm: it **aligns the training objective with the generation setting**, so that the model learns to produce a response given a prompt. But it also has a clear ceiling. If the training signal always comes from a **fixed reference distribution $q$**, the model can only be pushed toward $q$.

To break that ceiling, it is not enough to say “train longer” or “use better data.” The mathematical question is: **which parts of the SFT update are fixed by construction, and what extra degrees of freedom do we get if we relax them?**

The gradient makes this visible. The SFT loss is:

$$\mathcal{L}(\theta)=\mathbb{E}_{y\sim q}\Big[-\sum_t \log p_\theta(a_t\mid x,a_{<t})\Big]$$

Taking the gradient with respect to $\theta$ ($q$ does not depend on $\theta$, so the gradient passes through the expectation and the sum):

$$\nabla_\theta\mathcal{L}=-\,\mathbb{E}_{y\sim q}\Big[\sum_t \nabla_\theta\log p_\theta(a_t\mid x,a_{<t})\Big]$$

This gradient makes the SFT update explicit: every reference token contributes a term $\nabla_\theta\log p_\theta(a_t\mid\cdot)$. Under gradient descent, the update direction increases the probability of these reference tokens. More importantly, **the coefficient in front of every term is the same**: no weighting, and no negative sign.

<details class="math-note" markdown="1">
<summary><b>Why gradient descent increases the probability of a reference token</b></summary>

A single token contributes $\ell_t=-\log p_\theta(a_t)$ to the loss. Gradient descent moves the parameters in the direction that reduces the loss.

- Reducing $\ell_t=-\log p_\theta(a_t)$ is equivalent to increasing $\log p_\theta(a_t)$, which means increasing the probability $p_\theta(a_t)$.
- In terms of signs, this term's gradient is $-\nabla_\theta\log p_\theta(a_t)$. Gradient descent follows $-\nabla_\theta\mathcal{L}$, so this term becomes $+\nabla_\theta\log p_\theta(a_t)$.

The negative sign in $-\log$ cancels with the negative sign in gradient descent, so the update direction increases the probability of that token.

</details>

So if we want to treat different tokens differently, for example learning more from good answers and pushing bad answers down, we need to introduce a coefficient $w_t$ in front of each term:

$$\nabla_\theta\mathcal{L}=-\,\mathbb{E}_{y\sim q}\Big[\sum_t w_t\,\nabla_\theta\log p_\theta(a_t\mid x,a_{<t})\Big]$$

Ordinary SFT is the special case where **$w_t\equiv1$**. This 1 is not an extra term we added; it is the default weight already present in the original gradient.

From here, the path beyond SFT becomes clear: relax the things that were fixed. One is **the weight in front of each token**; the other is **where the samples come from**.

<div class="content-callout content-callout--insight">

  <div class="content-callout__row">
    <p class="content-callout__label"><strong>Fixed weights</strong></p>
    <p>SFT treats all reference tokens equally. The weight in front of every token is $1$, so the update can only increase the probability of those tokens. To express preference, the fixed $1$ has to become a variable weight $w_t$: good directions get larger weights, bad directions get smaller weights, or are even pushed down.</p>
  </div>

  <div class="content-callout__row">
    <p class="content-callout__label"><strong>Fixed sampling source</strong></p>
    <p>SFT samples come from the reference distribution $q$. If the training data always comes from $q$, the model is still only fitting that reference distribution. To let the model improve based on its own behavior, samples have to come from the model itself: $y\sim p_\theta$.</p>
  </div>

</div>

But letting the model generate its own samples is not enough. If it simply learns from whatever it generates, it is just imitating itself; there is no new learning signal. We still need to score those outputs, amplify the good directions, and suppress the bad ones.

Only when these two degrees of freedom are combined do we get the usual policy-gradient picture in RL [3]:

| Update weight / Sampling source | Sample $y\sim q$ (reference distribution) | Sample $y\sim p_\theta$ (model itself) |
|---|---|---|
| $w_t=1$ | SFT | Degenerate: self-imitation with no signal |
| $w_t=$ reward | Weighted behavior cloning: still limited by $q$ | Policy gradient: on-policy + reward |
{: .policy-matrix}

The difference between the bottom-right cell and SFT is not merely that we “added reward.” The objective has changed from **fitting the reference distribution $q$** to **making the model's own outputs receive higher reward**, i.e. optimizing $$\mathbb{E}_{y\sim p_\theta}[R]$$. Once the objective changes, the sampling distribution and the weights in front of the gradient change together.

## Conclusion

SFT is not mysterious. It is still next-token prediction; during training, we mask out the prompt and model the response conditionally. Written as an objective, it is fitting the reference distribution $q(y\mid x)$. In that sense, **SFT is behavior cloning**.

Its ceiling comes from the same place: in the gradient, every reference token has default weight $1$, so SFT can only increase the probability of reference tokens; and because the samples come from the reference distribution, the objective is also bounded by that distribution. Later methods such as policy gradient [3], PPO [4], DPO [5], and GRPO [6] may look different on the surface, but they all have to answer the same two questions: **where do the weights come from, and where do the samples come from?**

## References

<ol class="references-list">
  <li>Ouyang, Long, et al. “<a href="https://arxiv.org/abs/2203.02155">Training Language Models to Follow Instructions with Human Feedback</a>.” <em>arXiv</em>, 4 Mar. 2022.</li>
  <li>Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. “<a href="https://arxiv.org/abs/1503.02531">Distilling the Knowledge in a Neural Network</a>.” <em>arXiv</em>, 9 Mar. 2015.</li>
  <li>Sutton, Richard S., et al. “<a href="https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html">Policy Gradient Methods for Reinforcement Learning with Function Approximation</a>.” <em>Advances in Neural Information Processing Systems</em>, vol. 12, 1999.</li>
  <li>Schulman, John, et al. “<a href="https://arxiv.org/abs/1707.06347">Proximal Policy Optimization Algorithms</a>.” <em>arXiv</em>, 20 July 2017.</li>
  <li>Rafailov, Rafael, et al. “<a href="https://arxiv.org/abs/2305.18290">Direct Preference Optimization: Your Language Model Is Secretly a Reward Model</a>.” <em>arXiv</em>, 29 May 2023.</li>
  <li>Shao, Zhihong, et al. “<a href="https://arxiv.org/abs/2402.03300">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>.” <em>arXiv</em>, 5 Feb. 2024.</li>
</ol>
