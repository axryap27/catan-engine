# catan-engine

A fast C++ engine for 4-player Settlers of Catan, built to train a game-playing AI through self-play with PPO (reinforcement learning) and MCTS.

## Architecture

```
core      rules, game state, legal moves
belief    what each player knows about hidden cards
encode    game state -> neural network input
infer     runs the trained model from C++
selfplay  plays many games in parallel, writes training data
search    look-ahead search (evaluation only)
eval      tournaments and baselines
```

C++ handles the game, inference, and data generation; Python only trains the model. The two communicate through files: game data out, model checkpoints back.

Player-to-player trading is turned off for now; bank trades work normally.

## Status

Early development. Only `core` foundations exist so far (board topology, board generation, state types).

## Docs

- `DESIGN.MD`: system design and diagrams
- `OVERVIEW.MD`: prior work (Settlers-RL) and what we borrow or change from the implementation
