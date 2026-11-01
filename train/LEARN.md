# Learning RL on Catan

Read one section, then its worked exercise in `stage0.py`. To practice, delete a function's
body, rewrite it, and run its test.

```bash
.venv/bin/pip install -r train/requirements.txt                       # once
.venv/bin/python train/stage0.py                                      # watch each exercise learn
.venv/bin/python -m pytest train/tests/test_stage0.py -q              # every exercise
.venv/bin/python -m pytest train/tests/test_stage0.py -q -k exercise1  # just one
```

## 1. PyTorch in five ideas

- **Tensor**: a numpy array that PyTorch can differentiate (and move to a GPU). `x.shape`,
  `x.dtype`, `torch.zeros(3, 4)`. By convention the first dimension is the batch: 256 games
  in, 256 rows out.
- **Autograd**: a tensor made with `requires_grad=True` records every operation done to it.
  `loss.backward()` then computes d(loss)/d(tensor) for each one and stores it in `tensor.grad`.
- **Optimizer**: holds the parameters and moves each one against its gradient. `SGD` does
  `p -= lr * p.grad`; `Adam` adapts the step size per parameter and is what you'll use from
  now on. Gradients add up across `backward()` calls, so clear them with `opt.zero_grad()`.
- **nn.Module**: a network, i.e. layers with parameters. `policy(obs)` runs it,
  `policy.parameters()` is what the optimizer gets.
- **The training loop**, the same four lines in every exercise and in the Catan trainer:

  ```python
  loss = ...              # one number: how bad the parameters are on this batch
  opt.zero_grad()
  loss.backward()
  opt.step()
  ```

**Exercise 1**: fit a line by gradient descent.

## 2. RL vocabulary, in Catan terms

| Term | In Catan |
|---|---|
| agent, policy π | the network that picks moves |
| environment | the engine (`VecGames`) |
| observation s | the 2355 bytes one seat may see |
| action a | one legal `(type, arg)` |
| reward r | +1 for winning, at the end of the game |
| episode | one game, about 360 decisions (about 90 per seat) |

The goal: change the policy's parameters so the expected reward goes up.

## 3. A policy outputs scores, not moves

The network outputs one number per action, a **logit**. `softmax` turns logits into
probabilities (`exp` each one, divide by the sum), and the policy **samples** a move from
them. Sampling instead of always taking the top score is how the agent explores: a move it
rates low still gets tried sometimes, and may turn out good.

**Masking**: most moves are illegal at any moment. Set an illegal move's logit to `-inf`:
`exp(-inf) = 0`, so its probability is exactly 0, it is never sampled, and no gradient flows
into it.

**Exercise 2**: masked sampling.

## 4. Policy gradient (REINFORCE)

Supervised learning is told the right answer; RL is not. And the game isn't a formula, so you
can't backpropagate through it. The trick: sample a move, play it, see the reward, then

```python
loss = -(log_prob * advantage).mean()      # advantage = reward - baseline
```

Gradient descent on this raises the probability of moves that did better than the baseline
and lowers moves that did worse. The policy gradient theorem says that, averaged over many
samples, the gradient of this loss is exactly minus the gradient of the expected reward, so
lowering the loss raises the reward.

**Baseline**: subtracting the average reward doesn't change that average gradient but cuts its
noise a lot. Without it, every move in a winning game goes up, even the bad ones.

**Exercise 3**: REINFORCE on a slot machine with masked arms. Once it passes, print the mean
reward every 20 rounds, then try `advantage = reward` (no baseline) and compare how fast it
climbs.

## 5. Observation in, scores out

In Exercise 3 the best arm never changes, so the policy is 6 numbers. In Catan the best move
depends on the board, so the policy is a network: observation in, one logit per move out.

**Exercise 4**: the same learning rule with a network that has to read its input.

## 6. What changes for Catan

- **Two-part actions.** The network first picks a type (13 logits), then that type's argument
  from its section of the mask: a vertex for settlements and cities, an edge for roads, a tile
  for the robber, a card for `PLAY_DEV`, a relative seat for `STEAL`, a resource for discards,
  bank trades and the Year of Plenty / Monopoly picks. `ROLL`, `BUY_DEV` and `END_TURN` take no
  argument. The move's log-probability is the sum of both:
  `log π(type, arg) = log π(type) + log π(arg | type)`.
- **Delayed reward.** The only reward comes about 90 moves after the seat's first move. Which
  moves deserved it? Stages 3 to 5 (a value function, GAE, PPO) are all about answering that.
- **Four players.** Stage 6: one network plays every seat (self-play).

## Roadmap

| Stage | You build | Done when |
|---|---|---|
| 0 | these four exercises | `test_stage0.py` passes |
| 1 | the Catan policy: two-part masked sampling | a random network plays full games, never illegally |
| 2 | REINFORCE as seat 0 against 3 random bots | it beats the random bots |
| 3 | a value head (actor-critic) | learns faster than stage 2 |
| 4 | GAE | |
| 5 | PPO | stable training curve |
| 6 | self-play against past versions | it beats the greedy bot |
