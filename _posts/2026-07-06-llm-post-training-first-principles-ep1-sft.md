---
layout: post
title: "From SFT to RL: The Two Degrees of Freedom"
date: 2026-07-06 21:00:00 -0400
slug: "Post-Training from First Principles: SFT"
description: "How pretraining becomes conditional SFT, what limits a fixed-reference objective, and which parts of the SFT update later methods can relax."
tags: [LLM, Post-Training]
math: true
---

Many people, when first trying to understand Supervised Fine-Tuning (SFT), think of it as another training stage after pretraining: pretraining is self-supervised (SSL), while SFT is supervised. There is nothing wrong with this description, but I have always felt that there is a detail worth thinking through: **if SFT is still using cross-entropy loss to predict the next token, then how exactly is it different from SSL?**

This blog follows the learning signal from pretraining to SFT and then toward RL. First, the loss mask changes which tokens define the objective. Next, the reference responses determine the distribution that SFT fits. Finally, the gradient reveals two further choices: where training responses come from and what coefficients weight their token gradients.

## SFT is Language Modeling with a Mask

<p class="section-description">Pretraining and SFT use the same next-token model. The difference is which tokens receive loss and where those target tokens come from.</p>

Start with pretraining. Let $q_{\mathrm{pre}}$ denote the sequence distribution represented by the pretraining corpus, and let $z=(z_1,\ldots,z_L)$ be a sequence drawn from it. Maximum likelihood estimation (MLE) maximizes the expected log-probability of these sequences:

$$
\theta_{\mathrm{pre}}^*
=\arg\max_\theta
\mathbb{E}_{z\sim q_{\mathrm{pre}}}
\big[\log p_\theta(z)\big]
$$

Training is usually written as minimizing the equivalent negative log-likelihood:

$$
\mathcal{L}_{\mathrm{pre}}(\theta)
=\mathbb{E}_{z\sim q_{\mathrm{pre}}}
\big[-\log p_\theta(z)\big]
$$

An autoregressive language model factorizes the sequence probability into next-token probabilities:

$$
p_\theta(z)
=\prod_{t=1}^{L}
p_\theta(z_t\mid z_{<t})
$$

Substituting this factorization into the loss gives the pretraining objective used in practice:

$$
\mathcal{L}_{\mathrm{pre}}(\theta)
=\mathbb{E}_{z\sim q_{\mathrm{pre}}}
\left[
-\sum_{t=1}^{L}
\log p_\theta(z_t\mid z_{<t})
\right]
$$

**Pretraining therefore predicts every token in the sampled sequence.**

In pretraining, $z$ is treated as one undivided sequence. SFT organizes that same full sequence into two parts: a prompt prefix $x$ and a target response $y$. In other words, $z=(x,y)$, with $x$ provided as context and $y$ supplying the tokens the model should learn to generate. If $z$ were still trained with the pretraining objective unchanged, its loss would be $-\log p_\theta(x,y)$. We use lowercase $\ell$ for this single-example loss and uppercase $\mathcal{L}$ for its expectation over a data distribution.

The product rule splits the joint probability into the probability of the prompt and the conditional probability of the response:

$$
p_\theta(x,y)
=p_\theta(x)\,p_\theta(y\mid x)
$$

Taking the negative log exposes the two corresponding parts of the full-sequence loss:

$$
\begin{aligned}
\ell_{\mathrm{pre}}(x,y)
&=-\log p_\theta(x,y)
\\
&=\underbrace{-\log p_\theta(x)}_{\text{prompt loss}}
+\underbrace{-\log p_\theta(y\mid x)}_{\text{response loss}}
\end{aligned}
$$

Pretraining would optimize both terms. SFT keeps the same input sequence and the same autoregressive probabilities, but masks the token losses inside the prompt. At the sequence level, this removes the entire prompt term:

$$
\begin{aligned}
\ell_{\mathrm{SFT}}(x,y)
&=\underbrace{0\cdot\big[-\log p_\theta(x)\big]}_{\text{prompt loss masked}}
\\
&\quad+\underbrace{\big[-\log p_\theta(y\mid x)\big]}_{\text{response loss kept}}
\\
&=-\log p_\theta(y\mid x)
\end{aligned}
$$

**The model still reads the prompt, but the objective no longer asks it to predict the prompt. It only asks the model to predict the response conditioned on that prompt.**

For a response $y=(y_1,\ldots,y_T)$, its conditional log-probability expands over the response tokens. Treat the prompt $x$ as given, and let $q(\cdot\mid x)$ denote the distribution of reference responses for that prompt. The SFT loss is:

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
=\mathbb{E}_{y\sim q(\cdot\mid x)}
\left[
-\sum_{t=1}^{T}
\log p_\theta(y_t\mid x,y_{<t})
\right]
$$

The complete dataset objective also averages this conditional loss over prompts. We suppress that outer average here so the comparison with pretraining stays focused on the token-level objective.

<details class="math-note" markdown="1">
<summary><b>From Maximum Likelihood to the SFT Loss</b></summary>

Suppose the SFT dataset contains $N$ prompt-response pairs:

$$
\mathcal{D}_{\mathrm{SFT}}
=\big\{(x_i,y_i)\big\}_{i=1}^{N}
$$

① **Maximize the conditional likelihood.** The prompt $x_i$ is given, so each target contributes $p_\theta(y_i\mid x_i)$:

$$
\theta_{\mathrm{SFT}}^*
=\arg\max_\theta
\prod_{i=1}^{N}
p_\theta(y_i\mid x_i)
$$

② **Take the log.** The logarithm turns the product into a sum without changing the optimum:

$$
\theta_{\mathrm{SFT}}^*
=\arg\max_\theta
\sum_{i=1}^{N}
\log p_\theta(y_i\mid x_i)
$$

③ **Average over the dataset.** Dividing by $N$ does not change the optimum:

$$
\theta_{\mathrm{SFT}}^*
=\arg\max_\theta
\frac{1}{N}
\sum_{i=1}^{N}
\log p_\theta(y_i\mid x_i)
$$

④ **Turn maximization into minimization.** Adding a negative sign gives the empirical SFT loss:

$$
\widehat{\mathcal{L}}_{\mathrm{SFT}}(\theta)
=-\frac{1}{N}
\sum_{i=1}^{N}
\log p_\theta(y_i\mid x_i)
$$

⑤ **Write the population objective token by token.** The empirical average estimates an expectation over the data-generating distribution. With $x$ treated as given and $y\sim q(\cdot\mid x)$:

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
=\mathbb{E}_{y\sim q(\cdot\mid x)}
\left[
-\sum_{t=1}^{T}
\log p_\theta(y_t\mid x,y_{<t})
\right]
$$

</details>

The supervision does not come from a new loss function. It comes from **which response is supplied as the target**. In pretraining, the corpus supplies every next token. In SFT, a human, a stronger model, or a data pipeline supplies the response to imitate. The data source determines the targets, while the mask determines which tokens contribute to the loss [1].

**SFT does not replace language modeling. It changes the optimized likelihood from the full sequence to the response conditioned on the prompt.**


## SFT Fits the Reference Distribution

<p class="section-description">The SFT loss can also be read as a forward KL objective that fits the distribution behind the reference responses.</p>

The previous section derived the SFT loss from maximum likelihood. There is another way to read the same objective. Temporarily suppress the shared conditioning on a fixed prompt $x$. For any target distribution $q(y)$ and model distribution $p_\theta(y)$, their cross-entropy decomposes into the entropy of the target distribution plus a forward Kullback–Leibler (KL) divergence:

$$
\mathbb{E}_{y\sim q}
\big[-\log p_\theta(y)\big]
=
\underbrace{
\mathbb{E}_{y\sim q}
\big[-\log q(y)\big]
}_{\text{entropy of the target distribution}}
+
\underbrace{
\mathrm{KL}\big(q\,\|\,p_\theta\big)
}_{\text{forward KL}}
$$

<details class="math-note" markdown="1">
<summary><b>Why Cross-Entropy Equals Entropy Plus KL</b></summary>

Let $q(y)$ be the target distribution and $p_\theta(y)$ the model distribution.

① **Write the cross-entropy as a sum over possible outputs.**

$$
\mathbb{E}_{y\sim q}
\big[-\log p_\theta(y)\big]
=
\sum_y q(y)\big[-\log p_\theta(y)\big]
$$

② **Add and subtract $\log q(y)$ inside each term.**

$$
-\log p_\theta(y)
=-\log q(y)
+\log\frac{q(y)}{p_\theta(y)}
$$

③ **Substitute this identity into the sum and separate the two parts.**

$$
\begin{aligned}
\mathbb{E}_{y\sim q}
\big[-\log p_\theta(y)\big]
&=
\sum_y q(y)\big[-\log q(y)\big]
\\
&\quad+
\sum_y q(y)
\log\frac{q(y)}{p_\theta(y)}
\end{aligned}
$$

④ **Recognize the first sum as the entropy of $q$ and the second as the forward KL from $q$ to $p_\theta$.**

$$
\mathbb{E}_{y\sim q}
\big[-\log p_\theta(y)\big]
=
\mathbb{E}_{y\sim q}
\big[-\log q(y)\big]
+
\mathrm{KL}\big(q\,\|\,p_\theta\big)
$$

</details>

Now restore the conditioning on $x$. The reference distribution $q(\cdot\mid x)$ describes which responses appear in the SFT data for that prompt. It may be induced by human annotators, a stronger model, or a data-generation pipeline. Using the equivalent response-level form, the SFT loss from the previous section is the conditional cross-entropy between this reference distribution and the model:

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
=\mathbb{E}_{y\sim q(\cdot\mid x)}
\big[-\log p_\theta(y\mid x)\big]
$$

Applying the same decomposition gives:

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
=
\underbrace{
\mathbb{E}_{y\sim q(\cdot\mid x)}
\big[-\log q(y\mid x)\big]
}_{\text{conditional entropy of the reference responses}}
+
\underbrace{
\mathrm{KL}\big(
q(\cdot\mid x)\,\|\,p_\theta(\cdot\mid x)
\big)
}_{\text{forward KL}}
$$

The first term depends only on the reference distribution, not on the model parameters $\theta$. **Minimizing the SFT loss is therefore equivalent to minimizing the forward KL from $q(\cdot\mid x)$ to $p_\theta(\cdot\mid x)$.** In the ideal case, the two conditional distributions match.

The meaning is simple: **SFT fits the distribution of reference answers. Once that distribution is fixed, the likelihood objective contains no direct signal that prefers an output people would judge better than those references.** This is an objective-level ceiling, not a theorem that an SFT model can never outperform an annotator or teacher on a benchmark. Pretraining knowledge, generalization, and combining evidence across examples may all produce such gains. Better data can also improve SFT by changing $q(\cdot\mid x)$. What the fixed objective cannot do is decide how that reference distribution should improve.

This also explains the connection between SFT and knowledge distillation [2]. Both optimize cross-entropy against a reference distribution. The difference is where that distribution comes from. A teacher model gives distillation data, while human-written or filtered instruction data gives SFT data.

This perspective is also useful when reading experiments. Many papers use a stronger large model to generate data, then SFT a smaller model on that data, and finally report improvements on benchmarks. There may well be methodological contributions in such work, but the first thing to acknowledge is that **this pipeline is essentially learning from a teacher-induced reference distribution**. A natural baseline should be **“directly distill the same teacher.”** If a method cannot beat that baseline, the gain may be coming mostly from the data source, rather than from the new method itself.

## Breaking SFT's Ceiling

<p class="section-description">Once the reference distribution is fixed, the gradient reveals the two choices SFT cannot change: the update coefficients and the sampling source.</p>

The previous section identified the precise limitation of SFT. Training longer can fit the same reference distribution more closely, and better data can improve the model by changing that distribution. But once $q(\cdot\mid x)$ is fixed, the SFT objective contains no new signal about which outputs would be better than its references. The mathematical question is therefore: **which parts of the SFT update are fixed by construction, and what extra degrees of freedom appear when we relax them?**

The gradient makes this visible. The SFT loss is:

$$\mathcal{L}_{\mathrm{SFT}}(\theta)=\mathbb{E}_{y\sim q(\cdot\mid x)}\left[-\sum_{t=1}^{T} \log p_\theta(y_t\mid x,y_{<t})\right]$$

Taking the gradient with respect to $\theta$ ($q(\cdot\mid x)$ does not depend on $\theta$, so the gradient passes through the expectation and the sum):

$$\nabla_\theta\mathcal{L}_{\mathrm{SFT}}(\theta)=-\,\mathbb{E}_{y\sim q(\cdot\mid x)}\left[\sum_{t=1}^{T} \nabla_\theta\log p_\theta(y_t\mid x,y_{<t})\right]$$

This gradient makes the SFT update explicit. **Every observed target-token score term has the same coefficient $1$.** Under gradient descent, that term reinforces the observed token in its context. This does not mean every vocabulary probability increases. Softmax normalization lowers alternatives as the target probability rises. The important point is that SFT has no signed, token-dependent coefficient that says one observed behavior should be reinforced and another should be suppressed.

<details class="math-note" markdown="1">
<summary><b>What the Fixed Coefficient 1 Means</b></summary>

A single target token contributes:

$$
\ell_t
=-\log p_\theta(y_t\mid x,y_{<t})
$$

Its score term appears in the SFT gradient with coefficient $1$:

$$
\nabla_\theta \ell_t
=-\,1\cdot
\nabla_\theta\log p_\theta(y_t\mid x,y_{<t})
$$

The coefficient $1$ describes how the observed target score is weighted. It does not mean other tokens are unchanged. If $u_{t,k}$ is the logit of vocabulary token $k$, then:

$$
\frac{\partial \ell_t}{\partial u_{t,k}}
=
p_\theta(k\mid x,y_{<t})
-\mathbf{1}[k=y_t]
$$

Gradient descent raises the target token relative to its alternatives. The alternatives can decrease through the softmax normalization even though they do not appear as negatively weighted training examples.

</details>

To express relative quality directly, replace the fixed $1$ with a signed coefficient $w_t$:

$$\nabla_\theta\mathcal{L}_w(\theta)=-\,\mathbb{E}_{y\sim q(\cdot\mid x)}\left[\sum_{t=1}^{T} w_t\,\nabla_\theta\log p_\theta(y_t\mid x,y_{<t})\right]$$

Ordinary SFT is the special case where **$w_t\equiv1$**. A positive coefficient reinforces a sampled behavior, a negative coefficient suppresses it, and its magnitude controls the update strength.

From here, the path beyond SFT becomes clear: relax the things that were fixed. One is **the weight in front of each token**; the other is **where the samples come from**.

<div class="content-callout content-callout--insight">

  <div class="content-callout__row">
    <p class="content-callout__label"><strong>Fixed weights</strong></p>
    <p>Every observed target-token score term has coefficient $1$. To express relative quality, that fixed coefficient must become a signed weight $w_t$ that can reinforce or suppress a sampled behavior.</p>
  </div>

  <div class="content-callout__row">
    <p class="content-callout__label"><strong>Fixed sampling source</strong></p>
    <p>SFT samples come from $q(\cdot\mid x)$. To optimize the model based on its own behavior, responses instead have to be sampled from the current model: $y\sim p_\theta(\cdot\mid x)$.</p>
  </div>

</div>

Changing the sampling source alone is not enough. If the model simply trains on its own responses with the same coefficient $1$, it only imitates its current behavior and receives no new direction for improvement.

| Update coefficient / Sampling source | Sample $y\sim q(\cdot\mid x)$ | Sample $y\sim p_\theta(\cdot\mid x)$ |
|---|---|---|
| $w_t=1$ | SFT | Degenerate: self-imitation with no signal |
| Variable signed $w_t$ | Weighted behavior cloning | Both degrees of freedom are relaxed |
{: .policy-matrix}

This matrix is only a high-level map. It identifies the two parts of the SFT update that later methods may change, without specifying how any particular algorithm changes them.

The point of the bottom-right cell is only structural. **Moving beyond SFT may require changing both where training responses come from and how their token gradients are weighted.** How reinforcement learning constructs those weights is the subject of the next post.

## Conclusion

Pretraining and SFT optimize the same autoregressive model through token log-probabilities. Pretraining models every token in a corpus sequence $z$. SFT provides a prompt $x$, masks its token losses, and models only the response $y$. **SFT is therefore conditional behavior cloning:** it fits $p_\theta(\cdot\mid x)$ to the reference response distribution $q(\cdot\mid x)$.

That produces an objective-level ceiling, not an absolute bound on benchmark performance. Better data can move the reference distribution, and generalization can produce outputs that outperform individual references. But once $q(\cdot\mid x)$ is fixed, the SFT objective supplies no direct preference for outputs that are better than those references.

Methods beyond SFT may change the sampling source, the update coefficients, or both. These two questions—**where do the samples come from, and what coefficients weight their gradients?**—provide the high-level lens used in later posts to compare PPO [3], DPO [4], GRPO [5], and other post-training methods.

## References

<ol class="references-list">
  <li>Ouyang, Long, et al. “<a href="https://arxiv.org/abs/2203.02155">Training Language Models to Follow Instructions with Human Feedback</a>.” <em>arXiv</em>, 4 Mar. 2022.</li>
  <li>Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. “<a href="https://arxiv.org/abs/1503.02531">Distilling the Knowledge in a Neural Network</a>.” <em>arXiv</em>, 9 Mar. 2015.</li>
  <li>Schulman, John, et al. “<a href="https://arxiv.org/abs/1707.06347">Proximal Policy Optimization Algorithms</a>.” <em>arXiv</em>, 20 July 2017.</li>
  <li>Rafailov, Rafael, et al. “<a href="https://arxiv.org/abs/2305.18290">Direct Preference Optimization: Your Language Model Is Secretly a Reward Model</a>.” <em>arXiv</em>, 29 May 2023.</li>
  <li>Shao, Zhihong, et al. “<a href="https://arxiv.org/abs/2402.03300">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>.” <em>arXiv</em>, 5 Feb. 2024.</li>
</ol>
