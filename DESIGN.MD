# Catan Engine: System Design

A C++ engine for 4-player Settlers of Catan, plus a self-play pipeline that trains a neural network agent with PPO. Python is used only for the training loop. Why the design looks like this is explained in `OVERVIEW.MD`.

## Principles

1. **C++ everywhere except gradients.** Game logic, inference, search, and evaluation are C++. Python only trains the model.
2. **One loop for everything.** Self-play, search, evaluation, and human play all drive the engine through the same decision loop.
3. **State is plain data.** A game state is a small fixed-size struct (~250 bytes) that can be copied in nanoseconds, which search relies on.
4. **Public and hidden information are separate.** What everyone can see is kept apart from what only some players know, so hidden cards can be resampled.
5. **Randomness lives outside the state.** Dice and card draws are explicit chance events, so games replay exactly and search never sees the future.
6. **The action space never shrinks or reshapes.** Player trading is declared but disabled; enabling it later only adds outputs.

## System overview

```
 ┌──────────────────────────── C++ self-play process ────────────────────────────┐
 │                                                                               │
 │   stepping threads                          inference thread                  │
 │  ┌──────────────────┐   observations    ┌──────────────────────┐              │
 │  │ game slots       │ ────────────────► │ batch across games,  │              │
 │  │ (engine + belief │                   │ seats and opponent   │◄── weights   │
 │  │  + encoder)      │ ◄──────────────── │ versions; GPU model  │    (hot-swap)│
 │  └────────┬─────────┘   actions, values └──────────────────────┘      ▲       │
 │           │                                                           │       │
 └───────────┼───────────────────────────────────────────────────────────┼───────┘
             │ trajectory shards (files)                                 │
             ▼                                                           │
 ┌──────────────────────┐        new checkpoint + opponent pool          │
 │ Python PPO trainer   │ ───────────────────────────────────────────────┘
 └──────────┬───────────┘
            │ checkpoints
            ▼
 ┌──────────────────────┐
 │ eval harness (C++)   │  current model vs. past models and baselines
 └──────────────────────┘
```

The only link between C++ and Python is files: game data flows one way, model checkpoints flow back.

## Modules

```
                 ┌────────┐
                 │  core  │  rules, state, legal moves
                 └───┬────┘
                     │
                 ┌───▼────┐
                 │ belief │  what each player knows about hidden cards
                 └───┬────┘
                     │
                 ┌───▼────┐
                 │ encode │  state -> network input
                 └───┬────┘
             ┌───────┴────────┐        ┌───────────┐
         ┌───▼────┐      ┌────▼─────┐  │   infer   │  model runtime,
         │ search │      │ selfplay │◄─┤ Evaluator │  behind an interface
         └───┬────┘      └────┬─────┘  └───────────┘
             └───────┬────────┘
                 ┌───▼────┐
                 │  eval  │  tournaments, baselines, ratings
                 └────────┘

   bind: Python bindings for debugging and tests, never in training
```

| Module | Responsibility |
| --- | --- |
| `core` | Board, game state, rules, legal-move masks. Knows nothing about networks or players' knowledge |
| `belief` | Tracks, per player, which opponents' cards are known; samples plausible hidden hands |
| `encode` | Turns a state, from one player's point of view, into a network input |
| `infer` | Runs the exported model; other modules see only an `Evaluator` interface |
| `search` | Look-ahead search (ISMCTS) used at evaluation time |
| `selfplay` | Runs many games in parallel and writes training data |
| `eval` | Plays matches and measures strength |
| `bind` | Thin Python bindings for notebooks, debugging, and comparisons with other engines |

## The decision loop

The engine never "plays a turn." It reports who must decide what next, and applies the answer.

```
          ┌──────────────────────┐
     ┌───►│  next_decision(state)│
     │    └──────────┬───────────┘
     │               │
     │     ┌─────────┼──────────────┐
     │     ▼         ▼              ▼
     │  PLAYER     CHANCE        TERMINAL
     │  who, legal  dice, card    game over
     │  moves       draw, steal
     │     │         │
     │     ▼         ▼
     │  policy or   driver's
     │  search      seeded RNG
     │     │         │
     │     ▼         ▼
     │  ┌──────────────────────┐
     └──┤  apply(state, choice)│──► events (for beliefs and replays)
        └──────────────────────┘
```

- The deciding player is not always the player whose turn it is; after a 7, other players discard.
- Moves with only one legal option are applied automatically, without asking the network.
- Multi-part actions (Road Building, Year of Plenty, bank trades, discards) are broken into a sequence of single decisions.

## Game state

```
 GameState (~250 bytes, copied freely)
 ┌───────────────────────────────────┬──────────────────────────────┐
 │ PublicState                       │ HiddenState                  │
 │ board, buildings, roads, bank,    │ each player's resource cards │
 │ hand sizes, VP, robber, phase,    │ each player's dev cards      │
 │ turn counters                     │ dev card deck                │
 └───────────────────────────────────┴──────────────────────────────┘
               kept as-is                 resampled from beliefs
                                          when searching
```

The fixed board layout (which hexes touch which corners and edges) is built once and shared, never copied.

## Choosing an action

The network picks an action in parts: first a type, then its details.

```
 observation ──► shared trunk ──┬──► type head (13)       "build a settlement"
                                ├──► vertex head (54)     "on corner 31"
                                ├──► edge head (72)
                                ├──► tile head (19)
                                ├──► devcard head (5)
                                ├──► player head (3)
                                ├──► resource head (5)
                                └──► value head (4)       expected outcome per seat
```

Legal-move masks come from `core` with every decision, so the network can never choose an illegal move. All heads run in one GPU call.

## Training cycle

```
  play games ──► write data ──► train PPO ──► publish checkpoint ──┐
      ▲                                                           │
      └──────────── opponents drawn from past checkpoints ◄───────┘
```

- One or more seats play the current model; the rest play recent past versions.
- Games are cut into fixed-length chunks so training data is not skewed toward short games.
- Checkpoints save everything needed to resume exactly, not just the weights.

## Evaluation

- The current model plays fixed sets of boards and dice against past versions and simple baselines, rotating through every seat.
- Win rate is the headline number; 25% is break-even in a 4-player game.
- Look-ahead search is used only for milestone evaluations, because it is too slow for routine tracking.

## Testing

- Rule invariants checked after every move: card totals conserved, hand sizes match hands, legal-move lists never empty.
- Legal-move masks cross-checked against an independent, slow legality check and against another Catan engine (Catanatron).
- Beliefs checked against the true hidden state; resampled hands must look identical to the observer.
- Recorded games replay to the exact same final state.
