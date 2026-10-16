# Catan RL Engine: Overview and Prior Art

## What this project is

We are building an AI that plays the board game Settlers of Catan, trained by playing against copies of itself. The game engine is written in C++ for speed. The agent is a neural network trained with PPO (Proximal Policy Optimization), a standard reinforcement learning algorithm. The scope is the 4-player base game.

This document explains the closest prior work, what we borrow from it, and where we deliberately do something different. The engineering details live in `DESIGN.MD`.

## The prior work: Settlers-RL

[Settlers-RL](https://settlers-rl.github.io/) by Henry Charlesworth ([code on GitHub](https://github.com/henrycharlesworth/settlers_of_catan_RL)) is the closest existing project. It trained a PPO agent from scratch on full 4-player Catan, including trading between players.

It gives us a realistic sense of the cost. The run took roughly 3,500 training updates and 450 million game decisions, about a month on 32 CPU cores and one RTX 3090 GPU. The final agent was decent but weaker than a good human player.

The project's own conclusion was that trading between players should have been left out. Most of the trade offers the agent made were ones the other player could not accept. Two specific bugs caused that:

- The agent could ask for cards the other player did not have. Nothing restricted the "what I want in return" part of an offer to what the target actually held.
- Making an offer cost nothing, so the agent learned to fire off lopsided offers on the off chance one got accepted.

Both are fixable, but the simpler route is to avoid the problem. In our first phase, trading between players is turned off. The trade actions still exist in the action list so the network's shape does not have to change later; they are just never allowed. Trading with the bank (for example 4 of one resource for 1 of another) is unaffected.

## What we borrow unchanged

These choices from Settlers-RL are sound and cheap to copy.

| Idea | What it means | Why it works |
| --- | --- | --- |
| Board encoded per tile, with attention | Each of the 19 hexes becomes a 61-number feature vector. Two rounds of attention let every tile see every other tile before the results are combined | A tile's value depends on the rest of the board, such as nearby ports and who has built around it |
| LayerNorm | A normalization step after each attention and fully connected layer | Settlers-RL reports this mattered a lot for stable training |
| Number tokens one-hot | Each hex's dice number (2-12) is a category, not a quantity | 6 and 8 are the best numbers and 2 and 12 the worst, so "bigger is better" is false and the raw number would mislead the network |
| Seats relative to the viewer | Players are encoded as "me, next player clockwise, the one after, the one after that" | The same position looks the same no matter which seat the agent sits in |
| Card counts bucketed | Hand sizes are grouped as 0, 1, 2, 3, 4, 5, 6-7, 8+ | Holding 8 wood and holding 10 wood are strategically the same situation |
| Production vector | For each resource and each dice roll, how many cards the player receives (cities count double) | Summarizes the player's income directly instead of making the network work it out |
| Bounds on opponents' hands | For each opponent and resource, the fewest and most cards they could be holding, given everything visible | Exact and needs no learning |
| Fixed value normalization | The scale of predicted outcomes is fixed from a short trial run, not tracked with a running average | A running average gets distorted by old data as the agent improves |
| Recomputing advantages each epoch | The training targets are refreshed every pass over a batch, not computed once | A free improvement reported in a large study of PPO design choices (Andrychowicz et al.) |
| Masking illegal moves | Illegal actions get a large negative score before the network picks, so their probability is zero | The agent never wastes probability on moves it cannot make |

Two refinements to the last two rows:

- **Opponent bounds depend on who is looking.** If player A steals from player B, A and B both know which card moved, and nobody else does. So the bounds are tracked separately for each observer, in the `belief/` module described in `DESIGN.MD`. These bounds are our first, simple model of what opponents are holding. Anything more sophisticated has to beat them.
- **Use a large finite negative for masking, not negative infinity.** The common trick of adding `log(mask)` puts negative infinity into the scores, and the entropy calculation then hits `0 × -infinity`, which is NaN. Setting masked scores to something like -1e9 avoids that.

## What changes with trading turned off

**Fewer output heads.** The network chooses an action in parts, called heads. One head picks the action type ("build a road"), and others pick its details ("which edge"). Settlers-RL had ten heads. Three existed only for player trading (two that build the offered bundle of cards and one to accept or reject), so we have seven:

| Head | Choices | Used for |
| --- | --- | --- |
| type | 13 | which kind of action |
| vertex | 54 | where to build a settlement or city |
| edge | 72 | where to build a road |
| tile | 19 | where to move the robber |
| devcard | 5 | which development card to play |
| player | 3 | whom to steal from |
| resource | 5 | discards, Monopoly, Year of Plenty, bank trades |

The player head now only serves stealing, so it no longer needs to know the action type.

**Two-part choices become two decisions.** Some actions involve two picks that depend on each other. Year of Plenty takes two resources from the bank, but if the bank holds only one ore, taking two ore is illegal. Each head has its own legality mask, and two independent masks cannot express "ore is fine once but not twice." So Year of Plenty, bank trades (give, then receive) and Road Building (first road, then second) are each made as two separate decisions, the second one masked based on the first.

**The action list is fixed from day one.** All 13 action types are declared now, including the two trade types, which are always masked out. Existing heads never change size, so every saved model stays loadable. Turning trading on later means unmasking those types and adding new heads for the trade details, not resizing anything.

When that happens, the scores for the trade action types will have drifted to arbitrary values, because they never received any training signal while masked. Either reset those scores at the switch or briefly encourage the agent to explore the new actions. The rest of the network, which holds almost all the learning, carries over unchanged.

**Games get longer.** Without trading, players more often get stuck unable to build anything, so games run longer. That means:

- a turn cap is needed so every game ends;
- the discount factor stays at 0.999, Settlers-RL's value. It controls how much the agent cares about rewards far in the future, and our games got longer, not shorter.

## Where we deliberately differ

**The value head predicts all four players' outcomes, and all four are trained.** Besides choosing moves, the network estimates how well the game is going (the "value"). Settlers-RL predicted one number, for the agent's own seat. We want to add a look-ahead search later. In a 4-player game, each player in the search tree picks the move that is best for themselves, a rule called max^n, which needs an estimate for every seat. So the network outputs four values. Training only the agent's own seat would leave the other three as noise. Every recorded decision therefore trains all four, using each seat's actual final result.

**Development cards are counts, not a list.** Settlers-RL fed the list of held cards through its own small attention network. All cards of one type are identical, so a count per type (held, played, bought this turn) carries exactly the same information. It is simpler, cheaper and fixed in size.

**Less reward shaping.** Settlers-RL rewarded winning (+500) and also handed out small bonuses for victory points, buildings, robber moves, steals and development cards, plus a penalty for discarding, all balanced by hand. Every bonus like that is an opportunity for the agent to chase the bonus instead of the win. Settlers-RL saw the agent trade 4 ore to the bank for 1 ore, which is pointless except as a way to shrink its hand before a discard. We start with only the win/loss signal plus a reward for victory points gained, and treat the other bonuses as experiments to add later. A fast C++ engine generates enough games to make a sparser signal workable.

**Training, not game generation, is the bottleneck.** Settlers-RL collected 128,000 decisions per update (128 processes × 5 games × 200 decisions) and trained on each batch for 10 passes ("epochs"). In our setup:

- Producing one training decision costs about four network evaluations, one for each seat.
- Training on it for 10 epochs costs about 30, since each epoch runs the network forward and back.

So roughly 88% of GPU time goes to training. A faster engine mostly leaves the GPU queued up behind itself. Our response:

- Use 3-4 epochs instead of 10.
- Learn from every seat running the current network, not just one. That gives up to four times the data per game for free.
- Grow the batch only while training keeps up. A bigger batch means fewer updates for the same data: 450 million decisions in batches of 1 million is about 450 updates, against Settlers-RL's 3,500. Fewer updates usually means slower learning unless the learning rate is retuned.
- Measure the trainer before optimizing anything else.

**Simpler data collection.** Settlers-RL used a partly asynchronous setup, because each of its Python processes held four separate copies of the network and could not batch work onto the GPU. Our C++ engine runs one inference service that evaluates positions from all games and all seats in a single GPU batch, including opponents that use older versions of the network. That removes the reason for the asynchronous setup.

We do keep Settlers-RL's fixed-length chunks of 200 decisions per game. Waiting for whole games and stopping once a batch is full would collect a disproportionate number of short games and skew what the agent learns about how games end. Each chunk that stops mid-game uses the network's own value estimate for the rest of the game. Each decision also records which network version made it. Data from the previous version is still usable, since PPO can correct for one version of difference, and anything older is discarded.

That single GPU batch only stays efficient if few distinct opponent versions are playing at once. With 20 versions in play, each GPU call would handle only about 19 positions and spend most of its time on overhead. We limit it to about 4 opponent versions at a time and rotate them each update.

**Modern model export.** The trained network is handed to C++ with `torch.export` and AOTInductor, not TorchScript, which is no longer actively developed.

**Discards one card at a time, in a fixed order.** When a 7 is rolled, players with too many cards must discard half. Settlers-RL discarded one card per decision rather than choosing the whole set at once, though the project was unsure whether that was right. We are confident it is: it gives a search 5 options per step instead of a combinatorial explosion. We add one rule: within a single discard, the resource index can never decrease. Otherwise discarding four different resources could happen in 24 orders that all end in the same hand, which wastes search effort and splits the agent's probability across equivalent choices.

## Hyperparameters

Starting values, with Settlers-RL's for reference:

| Parameter | Settlers-RL | Ours |
| --- | --- | --- |
| Discount factor (gamma) | 0.999 | 0.999 |
| GAE lambda (advantage smoothing) | 0.95 | 0.95 |
| PPO clip epsilon | 0.2 | 0.2 |
| Epochs per batch | 10 | 3-4 |
| Minibatches per epoch | 64 | 64 |
| Entropy coefficient (exploration bonus) | 0.01 | 0.01 |
| Decisions per batch | 128,000 | 128,000, grown only while training keeps up |
| Chunk length (decisions) | 200 | 200 |

## Measuring progress

We copy Settlers-RL's evaluation method, with one addition. Every 100 updates, the current network plays 1,200 games against three copies of itself from 125 updates earlier, and separately against three copies from 250 updates earlier. In a 4-player game, a player no better than its opponents wins 25% of the time, so that is the baseline.

Our addition is fixed random seeds with every seat rotation, so both comparisons are played on the same boards with the same dice rolls. That removes a large source of luck.

Plot both curves. If the agent is really improving, it should beat the 250-update-old version by more than the 125-update-old one. Know how precise these numbers are: at a 25% win rate, 1,200 games give a margin of about ±2.5 percentage points, and the two curves must differ by about 3.5 points before the gap means anything. Late in training they often will not, so read the trend rather than single points. These evaluations do not use look-ahead search, which would be far too slow to run this often.

Average game length is also worth logging, since better players tend to finish faster. It is a weaker signal for us than for Settlers-RL, because the turn cap and the absence of trading both affect game length.

## Lessons from Settlers-RL's crash

Partway through the Settlers-RL run, the machine crashed. The network weights had been saved, but not the settings that controlled which past versions were chosen as opponents, and saved copies existed only every 25 updates. The opponent mix had to be rebuilt by guesswork. The win rate briefly dropped below the 25% baseline, and training never got back to its earlier pace of improvement.

So we save everything needed to resume exactly where we left off:

- the network weights;
- the opponent pool and its selection weights;
- the optimizer state;
- the value normalization settings;
- the update counter and random number generator state;
- the list of game data already used for training.

It is a few lines of code that protect weeks of compute.

## Realistic goals

A month of GPU time got Settlers-RL to below good-human strength. Our targets are to beat simple hand-written strategies and the published Settlers-RL agent, not human experts. Turning off trading and using a C++ engine should make training much faster, but reaching superhuman play would still take far more compute than we have.
