---
layout: post
title: "From SFT to RL: Reward and Policy Gradient"
date: 2026-07-17 21:00:00 -0400
slug: "Post-Training from First Principles: RLHF"
description: "Where RLHF's reward signal comes from, and how policy gradient turns a sequence-level score into token-level updates."
tags: [LLM, Post-Training]
math: true
---

Reinforcement learning (RL) is often presented as a list of algorithms: REINFORCE [1], PPO [2], DPO [3]. Before applying it to large language models (LLMs), it helps to return to RL's basic object: **a decision process with rewards**. A text prefix is the state $s_t$, the next token is the action $a_t$, and appending that token produces the next state. A complete response is a trajectory. LLM generation is therefore a Markov decision process (MDP) with text prefixes as states and almost deterministic transitions.

From this angle, RL is less a collection of algorithms than **a way to improve a policy from rewards observed on its trajectories**. The discussion follows two questions: where does the reward come from, and how does a reward assigned to a completed trajectory change the policy that generated it? Reward models, the advantage $A_t$, and the Kullback–Leibler (KL) constraint in reinforcement learning from human feedback (RLHF) all follow from those questions.

## Where Does Reward Come From?

<p class="section-description">RL begins with a question: what makes a response good? Before optimizing a policy, that judgment—the reward—must be specified.</p>

Start with the simplest case: **rewards can be verified directly.** Math and coding problems often have unambiguous answers. A verifier returns 1 for a correct answer and 0 otherwise. That is the reward $R$, supplied by the task itself, not by a learned model.

**The difficulty begins when no verifier exists.** There is no single correct answer to “Help me write an appropriate reply,” so the judgment must come from people. But an absolute score is unreliable: what does a 7 mean? Another annotator, or the same annotator at a later time, may use a different scale. **People are not especially good at maintaining a stable, shared scale of quality.**

**People are, however, usually more consistent at comparing two responses.** Given two responses to the same prompt, they can more reliably say which one is better. Data for open-ended tasks therefore often takes the form of a preference: for two responses $(y^+,y^-)$ to prompt $x$, an annotator says that $y^+$ is better than $y^-$. But **a preference is not yet a reward**. It ranks two responses, but gives neither a numeric score to an arbitrary response nor the size of the gap. Policy training still needs a scalar reward, so the question is how to construct a function $r(x,y)$ from many comparisons that can score any response while faithfully representing those preferences.

**The Bradley–Terry model turns relative judgments into scalar scores.** It assigns each response an unobserved strength $w$ and assumes [4]:

$$P(A\succ B)=\frac{w_A}{w_A+w_B}$$

Taking reward $r$ directly as $w$ causes a problem. A reward is unconstrained in sign and magnitude, while $w_A$ and $w_B$ must be positive for the right-hand side to be a valid probability. **The softmax parameterization $w=\exp(r)$ keeps $r$ unconstrained while ensuring $w>0$.** A comparison can then be written as a difference between two scores:

$$P(y^+\succ y^-\mid x)=\sigma\big(r(x,y^+)-r(x,y^-)\big)$$

<details class="math-note" markdown="1">
<summary><b>Bradley–Terry to Sigmoid</b></summary>

① **State the Bradley–Terry assumption.** Each response has a positive strength $w$, and the probability that A beats B depends on their relative strengths:

$$P(A\succ B)=\frac{w_A}{w_A+w_B}$$

② **Substitute the parameterization.** Replace the two strengths with $w_A=\exp(r_A)$ and $w_B=\exp(r_B)$:

$$
P(A\succ B)=\frac{\exp(r_A)}{\exp(r_A)+\exp(r_B)}
$$

③ **Rewrite the fraction as a difference.** Divide both numerator and denominator by $\exp(r_A)$:

$$P(A\succ B)=\frac{1}{1+\exp(r_B-r_A)}$$

④ **Recognize the sigmoid.** Since $r_B-r_A=-(r_A-r_B)$, this is exactly the definition of the sigmoid:

$$P(A\succ B)=\frac{1}{1+e^{-(r_A-r_B)}}=\sigma(r_A-r_B)$$

</details>

At this point, $r$ is still an abstract, unconstrained score. To learn it from preference data, **parameterize it as** $r_\phi(x,y)$, a function that maps a prompt and response to a scalar. This function is the **reward model (RM)**, and it must be **trained on preference data**. In common language-model RLHF setups, it is often initialized from a supervised model and given a randomly initialized linear scalar head [5]. This is a common design, not the only possible reward-model architecture.

Substitute $r_\phi$ into the Bradley–Terry probability. For a preference $(x,y^+,y^-)$, define:

$$r_\phi^+=r_\phi(x,y^+),\qquad r_\phi^-=r_\phi(x,y^-)$$

The model assigns probability $\sigma(r_\phi^+-r_\phi^-)$ to “$y^+$ is preferred to $y^-$.” **Training uses maximum likelihood estimation (MLE) to maximize the likelihood of the observed preference pairs.** Taking the log, negating it, and averaging over the dataset gives:

$$\mathcal{L}_{RM}(\phi)=-\,\mathbb{E}_{(x,y^+,y^-)\sim\mathcal{D}}\big[\log\sigma\big(r_\phi(x,y^+)-r_\phi(x,y^-)\big)\big]$$

<details class="math-note" markdown="1">
<summary><b>MLE Loss Derivation</b></summary>

① **Write the likelihood.** For $N$ preferences $\{(x_i,y_i^+,y_i^-)\}$, let $r_i^+=r_\phi(x_i,y_i^+)$ and $r_i^-=r_\phi(x_i,y_i^-)$. Assuming preferences are independent, the probability of the batch is the product of their probabilities:
$$L(\phi)=\prod_{i=1}^N \sigma\big(r_i^+-r_i^-\big)$$

② **Take the log.** The product becomes a sum. Since $\log$ is monotonic, maximizing $L$ is equivalent to maximizing $\log L$:
$$\log L(\phi)=\sum_{i=1}^N \log\sigma\big(r_i^+-r_i^-\big)$$

③ **Turn it into a loss.** Training conventionally minimizes an objective, so negate it:
$$-\log L(\phi)=-\sum_{i=1}^N \log\sigma\big(r_i^+-r_i^-\big)$$

④ **Average over the dataset.** Dividing by $N$ does not change the optimum and lets us write the result as an expectation:
$$\mathcal{L}_{RM}(\phi)=-\,\mathbb{E}_{(x,y^+,y^-)\sim\mathcal{D}}\big[\log\sigma(r_\phi^+-r_\phi^-)\big]$$

</details>

Gradient descent then fits $r_\phi$. If it learns stable patterns in human preferences rather than memorizing the training comparisons, it can generalize to unseen responses and score new outputs from the policy.

This completes the picture of where reward comes from: verifiable tasks provide rewards of 0 or 1 directly, while open-ended tasks use the learned $r_\phi$. But **$r_\phi$ approximates human preferences. It is not human preference itself.** A policy that optimizes it too aggressively may learn to exploit its gaps.


## Connecting a Trajectory-Level Reward to Each Step

<p class="section-description">Policy gradient answers one question: how can a reward for a complete response be combined with token-level gradients to update a policy?</p>

Model parameters directly control the log-probability of each generated token. A reward arrives only after the full response, with no ordinary autodiff path back to those parameters. **The question is how a trajectory-level reward should change the probability of each token.** This is the credit-assignment problem, and policy gradient is its answer [6].

One generation produces a trajectory $\tau$. Different samples yield different responses and rewards. The policy $\pi_\theta$ determines each trajectory's probability $P_\theta(\tau)$. The objective is not to reward one particular sample, but to **maximize the generation process's expected reward**:

$$J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R]$$

At this point, the internal structure of the trajectory does not matter. Let $R(\tau)$ be the reward assigned after sampling trajectory $\tau$. Once $\tau$ is fixed, $R(\tau)$ does not depend on $\theta$. The parameters only change the probability $P_\theta(\tau)$ of sampling that trajectory. Differentiating the expectation therefore gives:

$$\nabla_\theta J=\mathbb{E}_{\tau\sim\pi_\theta}\big[\nabla_\theta\log P_\theta(\tau)\,R\big]$$

<details class="math-note" markdown="1">
<summary><b>From Expected Reward to a Trajectory Gradient</b></summary>

① **Expand the objective.** Under the current policy, trajectory $\tau$ appears with probability $P_\theta(\tau)$. Expected reward is therefore a weighted sum over trajectories:

$$
J=\mathbb{E}_\tau[R]=\sum_\tau P_\theta(\tau)R
$$

② **Differentiate with respect to the parameters.** Reward $R$ comes from an external evaluation after sampling and does not directly depend on $\theta$. The gradient therefore acts only on the probability of each trajectory:

$$
\nabla_\theta J=\sum_\tau \nabla_\theta P_\theta(\tau)\,R
$$

③ **Rewrite the gradient in probability-weighted form.** This sum is not yet in a form that can be estimated by sampling, because $\nabla_\theta P_\theta(\tau)$ is not a probability-weighted sample term. Apply the score-function identity:

$$\nabla_\theta P_\theta(\tau)=P_\theta(\tau)\nabla_\theta\log P_\theta(\tau)$$

④ **Substitute and recognize the expectation:**

$$\nabla_\theta J=\sum_\tau P_\theta(\tau)\nabla_\theta\log P_\theta(\tau)R$$

$$\nabla_\theta J=\mathbb{E}_{\tau\sim\pi_\theta}\big[\nabla_\theta\log P_\theta(\tau)\,R\big]$$

</details>

The remaining term is $\nabla_\theta\log P_\theta(\tau)$, the gradient of a complete trajectory's probability. By itself, that is not yet useful for optimizing an LLM policy. An LLM does not produce a trajectory probability in one step. It produces a conditional probability for each next token. To turn the trajectory term into something the policy can optimize, $P_\theta(\tau)$ must be factored into the probabilities of its token-level decisions. **This is where the MDP view enters.** (The same factorization also follows directly from the autoregressive likelihood used in SFT.)

**For an autoregressive LLM, the trajectory log-gradient decomposes into a sum of token log-probability gradients:**

$$\nabla\log P_\theta(\tau)=\sum_t\nabla\log\pi_\theta(a_t\mid s_t)$$

<details class="math-note" markdown="1">
<summary><b>From a Trajectory Gradient to Token Gradients</b></summary>

① **Write the trajectory probability.** In an LLM, the action is the next token, and the transition simply appends that token to the current prefix. The next state is therefore determined by $(s_t,a_t)$, so $P(s_{t+1}\mid s_t,a_t)=1$. The trajectory probability reduces to the initial-state probability times the policy probability at each step:

$$
\begin{aligned}
P_\theta(\tau)
&= P(s_1)\prod_t P(a_t\mid s_t)\,
   \underbrace{P(s_{t+1}\mid s_t,a_t)}_{=1} \\
&= P(s_1)\prod_t\pi_\theta(a_t\mid s_t)
\end{aligned}
$$

② **Take the log and differentiate.** The initial-state probability $P(s_1)$ does not depend on $\theta$, leaving only one policy term for each token:

$$\nabla\log P_\theta=\sum_t\nabla\log\pi_\theta(a_t\mid s_t)$$

</details>

Substituting this token-level decomposition into the trajectory-level gradient gives the most naive policy gradient:

$$\nabla J=\mathbb{E}_\tau\Big[\sum_t R\,\nabla\log\pi_\theta(a_t\mid s_t)\Big]$$

This formula completes the connection from reward to token gradients. With the 0/1 verifier reward above, a successful trajectory raises the probability of every action it contains. An unsuccessful trajectory contributes zero gradient. The limitation is that every token in a successful trajectory receives the same signal, whether or not it mattered to the outcome.

## From Return to Advantage

<p class="section-description">The naive estimator gives every token the same terminal reward. Advantage estimates replace it with a context-dependent signal and reduce variance.</p>

**Every token receives the same terminal reward, regardless of how much it contributed to the final outcome.** A late token that determines whether an answer is correct and an early token that has no effect receive the same weight. For any one token, much of the reward may be determined by later sampled tokens. Those fluctuations become noise in its gradient, producing high variance.

At time $t$, write the return as $R=G_{<t}+G_t$. For the terminal-only rewards considered so far, $G_{<t}\equiv0$, so removing it does no work yet. The split becomes useful once rewards include intermediate terms, such as the per-token KL penalty introduced later. In general, the past term has already happened and cannot change with the current choice of $a_t$, so it should not affect this update.

The distinction between $G_t$ and $V(s_t)$ matters. **$G_t$ is the return actually observed after taking $a_t$ and sampling the rest of the trajectory.** It is a Monte Carlo sample of the action value:

$$\mathbb{E}[G_t\mid s_t,a_t]=Q_\pi(s_t,a_t)$$

**$V(s_t)$ is the expected return before choosing an action.** It averages those action values over the policy's possible actions:

$$V_\pi(s_t)=\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}[Q_\pi(s_t,a_t)]$$

So $G_t-V(s_t)$ asks whether this sampled continuation, after taking this particular action, did better or worse than what the policy expected from the same prefix.

<details class="math-note" markdown="1">
<summary><b>Why the Past Return and State Baseline Can Be Removed</b></summary>

① **State the baseline lemma.** Any term $b(s_t)$ that depends only on the state, not the current action, contributes zero in expectation to the policy gradient:

$$\mathbb{E}_{a_t\sim\pi}[\nabla\log\pi(a_t\mid s_t)\cdot b(s_t)]=0$$

② **Verify the lemma.** Rewriting $\nabla\log\pi$ as $\nabla\pi/\pi$ leaves the gradient of the sum of policy probabilities:

$$
\begin{aligned}
&\mathbb{E}_{a_t\sim\pi}[\nabla\log\pi(a_t\mid s_t)\cdot b(s_t)] \\
&= b(s_t)\sum_{a_t}\pi(a_t\mid s_t)\nabla\log\pi(a_t\mid s_t) \\
&= b(s_t)\sum_{a_t}\nabla\pi(a_t\mid s_t) \\
&= b(s_t)\nabla\sum_{a_t}\pi(a_t\mid s_t) \\
&= 0
\end{aligned}
$$

③ **Apply it to the return.** Given $s_t$, $G_{<t}$ is independent of the current $a_t$ and can be removed. For terminal-only reward, it is already zero. $V(s_t)$ also depends only on the state, so subtracting it does not change the expected gradient. The weight becomes the advantage:

$$
\begin{aligned}
\nabla J
&= \mathbb{E}_\tau\Big[\sum_t \big(G_t-V(s_t)\big)
   \,\nabla\log\pi_\theta(a_t\mid s_t)\Big] \\
A_t &:= G_t-V(s_t)
\end{aligned}
$$

</details>

The LLM policy gradient can therefore be written as:

$$
\begin{aligned}
\nabla_\theta J
&= \mathbb{E}_{\tau\sim\pi_\theta}\Big[
   \sum_t A_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big] \\
A_t &:= G_t-V(s_t)
\end{aligned}
$$

Each token $a_t$ is updated in the direction that raises its probability, $\nabla\log\pi(a_t\mid s_t)$. The weight $A_t$ is called the advantage. **It measures whether the return after this token was above or below the baseline for its context, and by how much.**

<details class="math-note" markdown="1">
<summary><b>How to Read $A_t$</b></summary>

① **Read the direction and magnitude.** $A_t$ determines the direction and strength of this token's update. When $A_t>0$, raise the probability of $a_t$ after prefix $s_t$. When $A_t<0$, lower it. Its absolute value sets the strength. When $A_t=0$, this term contributes no gradient.

② **Do not treat it as exact attribution.** One update adds all $A_t\nabla\log\pi(a_t\mid s_t)$ terms and updates the shared parameters $\theta$. Each term carries its own prefix $s_t$, so it means “adjust $a_t$ in this context,” not “always prefer or avoid this token.”

③ **Average out noise across samples.** A single rollout is noisy. Across many rollouts, directions that recur consistently remain, while conflicting signals cancel. **Credit assignment emerges statistically across rollouts, not from a precise attribution within one trajectory.**

</details>

## Putting Reward into Policy Gradient

<p class="section-description">Policy gradient only requires one scalar reward per trajectory. The source of that reward determines what the policy optimizes and where it can fail.</p>

Policy gradient does not prescribe where reward comes from. For verifiable tasks, use the checker's 0/1 output directly as $R$. For open-ended tasks, substitute the reward-model score $r_\phi(x,y)$ for $R$ and follow the same policy gradient.

From here on, fix a prompt $x$ and let $y=(a_1,\ldots,a_T)$ denote the complete sampled response. In the MDP notation above, $y$ is the generated part of the trajectory $\tau$. **Because an LLM transition only appends each new token to the prefix, $y$ and $\tau$ are equivalent once $x$ is fixed.**

Substituting $R=r_\phi(x,y)$ into the token-level policy gradient gives:

$$
\nabla_\theta J_x
=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}
\left[
r_\phi(x,y)
\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right]
$$

After fitting $r_\phi$, it is typically held fixed while a separate policy is optimized against it. The policy no longer sees human judgments. **It is rewarded only for a higher reward-model score.**

## Reward Hacking and KL Constraint

<p class="section-description">A learned reward is only a proxy for quality. The KL constraint limits how far policy optimization can move from the reference model.</p>

**But the reward model is fitted to finite preference data. It is not quality itself.** Away from the training data and the reference policy $\pi_{SFT}$, its score can diverge from what people would actually prefer. The policy may then find low-quality responses that exploit those errors for inflated scores. This is **reward hacking**. Unnecessarily lengthening a response is one common example (longer answers often look preferable, so the reward model may learn length rather than quality). **Standard RLHF therefore adds a Kullback–Leibler (KL) constraint to the objective**, keeping the policy near $\pi_{SFT}$ [5, 7]:

$$
J_x(\theta)=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[r_\phi(x,y)]
-\beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{ref}(\cdot\mid x)\big)
$$

Recall the original objective:

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R]
$$

The first term in $J_x(\theta)$ is the same objective with the reward-model score $r_\phi(x,y)$ in place of $R$. RLHF adds the second term, which penalizes the distance between the policy and the reference distribution. The goal therefore changes from maximizing expected reward alone to maximizing expected reward while limiting how far the policy can move from $\pi_{ref}$.

Here, $\pi_{ref}=\pi_{SFT}$ is a frozen reference policy. Because the current policy appears first and the reference policy second, this direction is commonly called the **reverse KL**. It compares the two response distributions as follows:

$$
\mathrm{KL}\big(\pi_\theta\|\pi_{ref}\big)
=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}
\left[
\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}
\right]
$$

Substituting this definition into $J_x(\theta)$ puts both terms under one expectation:

$$
J_x(\theta)
=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}
\left[
r_\phi(x,y)
-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}
\right]
$$

This form makes the sampled signal concrete. A response receives its reward-model score, then pays a penalty when the current policy makes it more likely than the reference policy does.

Directly differentiating this objective gives:

$$
\nabla_\theta J_x
=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}
\left[
\left(
r_\phi(x,y)
-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}
\right)
\nabla_\theta\log\pi_\theta(y\mid x)
\right]
$$

<details class="math-note" markdown="1">
<summary><b>How the KL Penalty Enters the Policy Gradient</b></summary>

Unlike the fixed reward model $r_\phi$, the KL contains the policy being optimized. Its gradient therefore has to be worked out directly.

① **Write the KL as a sum over responses.** For a fixed prompt $x$:

$$
\mathrm{KL}(\pi_\theta\|\pi_{ref})
=\sum_y \pi_\theta(y\mid x)
\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}
$$

The policy appears twice: once as the probability assigned to $y$, and again inside the log-ratio.

② **Apply the product rule.** Differentiating these two appearances produces two terms:

$$
\begin{aligned}
\nabla_\theta\mathrm{KL}
&=\sum_y
\nabla_\theta\pi_\theta(y\mid x)
\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}
\\
&\quad+\sum_y
\pi_\theta(y\mid x)
\nabla_\theta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}
\end{aligned}
$$

The reference policy is frozen, so the derivative inside the second sum is simply $\nabla_\theta\log\pi_\theta(y\mid x)$.

③ **The second sum is zero.** Using $\pi_\theta\nabla_\theta\log\pi_\theta=\nabla_\theta\pi_\theta$, it becomes the gradient of the total probability assigned to all responses:

$$
\begin{aligned}
\sum_y\pi_\theta(y\mid x)
\nabla_\theta\log\pi_\theta(y\mid x)
&=\sum_y\nabla_\theta\pi_\theta(y\mid x)
\\
&=\nabla_\theta\sum_y\pi_\theta(y\mid x)
\\
&=\nabla_\theta 1
\\
&=0
\end{aligned}
$$

④ **Rewrite the remaining sum as an expectation.** Apply $\nabla_\theta\pi_\theta=\pi_\theta\nabla_\theta\log\pi_\theta$ to turn the remaining term into a probability-weighted sum. That sum is exactly an expectation under $\pi_\theta$:

$$
\begin{aligned}
\nabla_\theta\mathrm{KL}(\pi_\theta\|\pi_{ref})
&=\sum_y \pi_\theta(y\mid x)
\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}
\nabla_\theta\log\pi_\theta(y\mid x)
\\
&=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}
\left[
\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}
\nabla_\theta\log\pi_\theta(y\mid x)
\right]
\end{aligned}
$$

⑤ **Decompose the response-level penalty.** An autoregressive LLM assigns a response its probability by multiplying the conditional probability of every generated token. The current and reference policies therefore factorize as:

$$
\pi_\theta(y\mid x)
=\prod_t\pi_\theta(a_t\mid s_t),
\qquad
\pi_{ref}(y\mid x)
=\prod_t\pi_{ref}(a_t\mid s_t)
$$

Their ratio is a product of token-level ratios:

$$
\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}
=\prod_t
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{ref}(a_t\mid s_t)}
$$

Taking the log turns that product into a sum:

$$
\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}
=\sum_t
\log\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{ref}(a_t\mid s_t)}
$$

The reward-model score arrives at the end of the response, while the KL penalty contributes a term at every generated token.

</details>

## Conclusion

[SFT](https://shawnyin128.github.io/blogs/post-training-from-first-principles-sft) and policy gradient can now be placed in the same frame. Written side by side, they share **the same optimization skeleton**:

$$\mathcal{L}=-\,\mathbb{E}\Big[\sum_t w_t\,\log\pi_\theta(a_t\mid s_t)\Big]\quad(\pm\ \beta\,\text{KL})$$

In SFT, $w_t\equiv1$ and samples come from an offline demonstration dataset, so training can only increase the probability of demonstrated tokens. In RL, $w_t=A_t$, the advantage computed from reward. Samples come from the model itself through on-policy sampling. RL can reward as well as penalize, and may exceed its demonstrations.

Put differently, **SFT is the special case with a constant weight of 1 and offline demonstrations.** RL lets reward determine the weight and draws samples from the current policy. PPO makes this optimization stable. DPO rewrites the KL-regularized objective directly as a preference objective, avoiding a reward model. GRPO turns to verifiable rewards and removes the value network. All of them build on the relationship between reward and policy gradient developed here.

## References

<ol class="references-list">
  <li>Williams, Ronald J. “<a href="https://doi.org/10.1007/BF00992696">Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning</a>.” <em>Machine Learning</em>, vol. 8, 1992, pp. 229–256.</li>
  <li>Schulman, John, et al. “<a href="https://arxiv.org/abs/1707.06347">Proximal Policy Optimization Algorithms</a>.” <em>arXiv</em>, 20 July 2017.</li>
  <li>Rafailov, Rafael, et al. “<a href="https://arxiv.org/abs/2305.18290">Direct Preference Optimization: Your Language Model Is Secretly a Reward Model</a>.” <em>arXiv</em>, 29 May 2023.</li>
  <li>Bradley, Ralph Allan, and Milton E. Terry. “<a href="https://doi.org/10.1093/biomet/39.3-4.324">Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons</a>.” <em>Biometrika</em>, vol. 39, no. 3/4, 1952, pp. 324–345.</li>
  <li>Stiennon, Nisan, et al. “<a href="https://arxiv.org/abs/2009.01325">Learning to Summarize from Human Feedback</a>.” <em>arXiv</em>, 2 Sept. 2020.</li>
  <li>Sutton, Richard S., et al. “<a href="https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html">Policy Gradient Methods for Reinforcement Learning with Function Approximation</a>.” <em>Advances in Neural Information Processing Systems</em>, vol. 12, 1999.</li>
  <li>Ziegler, Daniel M., et al. “<a href="https://arxiv.org/abs/1909.08593">Fine-Tuning Language Models from Human Preferences</a>.” <em>arXiv</em>, 18 Sept. 2019.</li>
</ol>
