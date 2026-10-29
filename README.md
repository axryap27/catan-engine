# catan-engine

A fast C++20 engine for 4-player Settlers of Catan, built to train a game-playing AI through self-play with PPO (reinforcement learning) and (maybe?!) MCTS.

Player-to-player trading is turned off for now; bank and port trades work normally.

## Performance

Apple M3 Pro (6 performance + 6 efficiency cores), native arm64 Release build:

| Metric | Random play | Greedy bot play |
|---|---|---|
| Throughput, 1 thread | 8.3M decisions/s | 1.8M decisions/s |
| Throughput, 12 threads | **72M decisions/s, 43k games/s** | 15M decisions/s, 33k games/s |
| Scaling, 4 / 8 threads | 3.87x / 6.80x | 3.88x / 6.82x |
| Decision latency p50 / p99 / p99.9 | ≤42 ns / 2.3 µs / 5.2 µs | 458 ns / 2.2 µs / 2.8 µs |
| Encoding an observation + mask, p50 / p99 | 250 ns / 375 ns | 250 ns / 417 ns |

A decision is one call to find the legal moves, choose one, and apply it. Dice rolls, card draws, and steals count as decisions too. Random play gives long games (about 340 turns); greedy play gives realistic ones (about 78 turns, p99 372 µs per game). Replays are deterministic and do not depend on the number of threads or the CPU architecture (arm64 and x86-64 give identical games). My Mac clock ticks every 41 ns, which _limits latency precision_.

## Getting started

Requires CMake 3.24+, Ninja, and a C++20 compiler (Clang or GCC 14). On Apple Silicon with an Intel Homebrew CMake (`/usr/local/bin/cmake`), add `-DCMAKE_OSX_ARCHITECTURES=arm64` to the configure step, or the binaries run under Rosetta at about two-thirds speed.

```bash
cmake --preset release && cmake --build --preset release

build/release/tools/catan_sim 42                        # play one random game, print every event
build/release/tools/catan_match greedy random 250       # match over 250 seeds, win rates with 95% CIs
build/release/tools/catan_bench 50000 --sweep           # throughput and scaling across threads
build/release/tools/catan_bench 20000 --policy greedy --latency   # per-decision latency percentiles
build/release/tools/catan_bench 20000 --encode --latency          # adds belief tracking + encoding
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
encode    game state -> neural network input                    done
bind      Python bindings and a text view of the board
infer     runs the trained model from C++
selfplay  plays many games in parallel, writes training data
search    look-ahead search (evaluation only)
```

## What's left

The game engine, hidden-card tracking, baselines, and the network's input encoding are done. TODO: add Python bindings, build the PPO trainer, then build the C++ self-play and inference pipeline.

## Docs

- `DESIGN.MD`: system design and diagrams
- `OVERVIEW.MD`: prior work (Settlers-RL) and what we borrow or change
