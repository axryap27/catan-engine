# Catan RL Engine: Design Notes from Settlers-RL

## Source

These notes map Henry Charlesworth's [Settlers-RL](https://settlers-rl.github.io/) onto our C++ engine with trading disabled. It is the closest prior art: full 4-player Catan with trading, PPO from scratch, [code on GitHub](https://github.com/henrycharlesworth/settlers_of_catan_RL).

His result sets the calibration. Roughly 3,500 updates and 450 million decisions, about a month on 32 cores plus a GTX 3090, ending below good-human strength. He concluded he would remove trading if starting over, because agents wasted most proposals on trades the target could not accept.

Two specific bugs produced that outcome, both fixable: the receive side of the trade head was never masked against the target's actual holdings, and proposing cost nothing, so unfair offers were free lottery tickets. Our phase 1 sidesteps both by disabling trading entirely while keeping the rules and action slots in place.

## Adopt as-is

These choices are sound and cheap to replicate. Take them without re-deriving.

| Component | Detail | Why |
| --- | --- | --- |
| Tile module | 61-dim features per tile, 19 tiles, two rounds of multi-head attention with a shared-weight FC layer between, then concatenate and project | Tile value depends on the whole board, so global context is needed |
| LayerNorm | After both the FC and attention layers | He reports it mattered a lot |
| Token encoding | One-hot the dice token, never the raw number | 6 and 8 are best, 2 and 12 worst, so there is no useful ordering |
| Player indexing | Relative: self, CW+1, CW+2, CW+3 | Free seat invariance |
| Resource counts | Bucketed 1, 2, 3, 4, 5, 6-8, 8+ | 8 wood and 10 wood are the same situation |
| Production vector | Per resource, settlements paying out on each dice value, cities counting double | Raw counts, not one-hot |
| Dev cards | Embedding, then attention, then sum over the list | Handles variable held and played counts |
| Opponent beliefs | Min and max possible holdings per resource from public info | Exact interval bounds, no learning required |
| Value normalization | Fixed mean and std from a short pilot run | Beats a running average, which stale entries corrupt as the agent improves |
| Return recomputation | Recompute GAE returns after each epoch | Free performance per the Andrychowicz study |
| Masking | Add log(mask) to logits before softmax | Valid stays 0, invalid goes to negative infinity |

The min/max interval tracker lives in `core` and updates inside `apply()`, not at encode time. It is the belief layer for phase 1 and the baseline any learned model has to beat.

## What drops without trading

Ten action heads become seven. The two recurrent trade-bundle heads and the accept/reject head go away, which removes the single hardest part of his architecture and all of its observed failure.

The choose-player head survives but serves only stealing, so it no longer needs the action type as input.

Action types stay at 13 in the declaration. `PROPOSE_TRADE` and `RESPOND_TRADE` are permanently masked, never reachable. This is non-negotiable: growing the action space later would invalidate every checkpoint, so the space is fixed now and the curriculum becomes a mask schedule.

Two second-order effects:

- Games run longer. Resource deadlock is more common without trading, so turn counts rise.
- Keep gamma at 0.999 rather than lowering it. His value was tuned for games with trading, and our horizon got longer, not shorter.

When trading is unmasked later, the masked-out logits will have drifted to arbitrary values since they received no gradient. Either reinitialize those output rows at the transition or add a short entropy bonus on the newly unmasked actions. The trunk and value head transfer cleanly, which is where almost all the learning lives.

## Where to deviate

**Value head outputs 4 values, not 1.** He trains one agent against frozen opponents, so a scalar suffices. ISMCTS beyond the root in a 4-player game needs max^n backup, which requires each seat's value. Build the 4-vector now even though phase 1 reads one entry.

**Less reward shaping.** He uses +500 for a win plus small dense rewards for VP, buildings, robber moves, steals and dev cards, with a discard penalty, all balanced by feel. That is a lot of hand-tuning and it is where reward hacking enters. He observes the agent exchanging 4 stone for 1 stone to dump cards, which smells like a shaping artifact.

Start with win/loss plus VP gain only. Treat the rest as an ablation. C++ throughput buys the sparser signal he could not afford.

**Scale batch size, not update count.** His batch was 128 processes x 5 games x 200 decisions = 128,000 decisions per update, with 10 epochs and 64 minibatches of 2,000. Keep those ratios. Our decisions per second will be much higher, so grow the batch rather than running more updates on small ones.

**Semi-async becomes unnecessary.** He went semi-asynchronous because each Python process held four separate policy networks and could not batch on GPU. With TorchScript inference in C++ there is one engine batching across all games and all seats at once, including opponent policies, which are just different weight sets. Run closer to synchronous with a refill queue. Simpler and faster.

**Keep his discard head.** One resource per decision rather than a bundle. He is unsure about it; we should not be. Single-resource discards give each search node 5 branches instead of a combinatorial fan-out.

## Evaluation and calibration

Copy his eval protocol exactly. Every 100 updates, play 1,200 games with the current policy against three copies of the policy from 125 updates earlier, and separately against 250 updates earlier. Baseline is 25%.

Plot both lags. The 250-lag curve sitting above the 125-lag curve is the sanity check that learning is real rather than noise.

Also log average decisions per game across all four players. Declining game length is a reasonable proxy for improvement and costs nothing to record.

His hyperparameters, as a starting point:

| Parameter | Value |
| --- | --- |
| Discount (gamma) | 0.999 |
| GAE lambda | 0.95 |
| PPO clip epsilon | 0.2 |
| Epochs per batch | 10 |
| Minibatches per epoch | 64 |
| Entropy coefficient | 0.01 |
| Batch size (decisions) | 128,000 |

### The lesson from his accident

His machine crashed mid-run. He had checkpointed policy weights but not the opponent-pool sampling distribution, and only had copies every 25 updates. He had to reconstruct an ad-hoc distribution, the win rate briefly fell below the 25% baseline, and the recovery never regained the original rate of improvement.

Checkpoint the opponent pool state alongside the weights. This is a few lines of serialization that protects weeks of compute.

### Expectation setting

A month of 3090 time got him below good-human. Target beating the heuristic baselines and his published agent, not humans. With trading off and a C++ engine the throughput should be much better, but the compute gap to anything superhuman remains large.