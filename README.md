# catan-engine

A fast C++20 engine for 4-player Settlers of Catan, built to train a game-playing AI through self-play with PPO (reinforcement learning) and (maybe?!) MCTS.

Player-to-player trading is turned off for now; bank and port trades work normally.

## Performance

Apple M3 Pro (6 performance + 6 efficiency cores), Release build:

| Metric | Random play | Greedy bot play |
|---|---|---|
| Throughput, 1 thread | 5.4M decisions/s | 1.3M decisions/s |
| Throughput, 12 threads | **47M decisions/s, 28k games/s** | 9.1M decisions/s, 19.8k games/s |
| Scaling, 4 threads | 3.88x | 3.76x |
| Decision latency p50 / p99 / p99.9 | ≤43 ns / 3.75 µs / 9.0 µs | 584 ns / 3.4 µs / 5.9 µs |

A decision is one call to find the legal moves, choose one, and apply it. Dice rolls, card draws, and steals count as decisions too. Random play gives long games (about 340 turns); greedy play gives realistic ones (about 78 turns, p99 546 µs per game). Replays are deterministic and do not depend on the number of threads. My Mac clock ticks every 41 ns, which _limits latency precision_.

## Getting started

Requires CMake 3.24+, Ninja, and a C++20 compiler (Clang or GCC 14).

```bash
cmake --preset release && cmake --build --preset release

build/release/tools/catan_sim 42                        # play one random game, print every event
build/release/tools/catan_match greedy random 250       # match over 250 seeds, win rates with 95% CIs
build/release/tools/catan_bench 50000 --sweep           # throughput and scaling across threads
build/release/tools/catan_bench 20000 --policy greedy --latency   # per-decision latency percentiles
```

Tests:

```bash
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
cmake --preset asan && cmake --build --preset asan && ctest --preset asan   # sanitizers + invariant checks
```

CI runs the tests with GCC and Clang on Linux and macOS, under ASan/UBSan, plus a 10,000-game fuzz that cross-checks every legal move against an independent reference implementation.

## Architecture

```
core      rules, game state, legal moves                       done
belief    what each player knows about hidden cards             done
eval      baselines (random, greedy) and paired-seed matches    done
encode    game state -> neural network input
bind      Python bindings and a text view of the board
infer     runs the trained model from C++
selfplay  plays many games in parallel, writes training data
search    look-ahead search (evaluation only)
```

## What's left

The game engine, hidden-card tracking, and baselines are done. TODO: turn the game state into network input, add Python bindings, build the PPO trainer, then build the C++ self-play and inference pipeline.

## Docs

- `DESIGN.MD`: system design and diagrams
- `OVERVIEW.MD`: prior work (Settlers-RL) and what we borrow or change
