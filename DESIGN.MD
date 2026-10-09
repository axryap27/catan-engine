# Catan Engine: System Design

Last updated: 2026-10-09

Scope: 4-player base game, C++ engine, PPO self-play, trading implemented in the rules layer but disabled for the agent. Companion doc: `settlers-rl-notes.md`.

## Guiding constraints

1. Everything in C++ except gradient computation. The only Python is the model definition, the training loop, and experiment logging.
2. The action space is fixed now and never grows. Trading actions are declared and permanently masked. Adding them later must not change any tensor shape.
3. State is a POD you can `memcpy`. Search clones it constantly.
4. Public and hidden state are separate structs. Determinization resamples one and keeps the other.

## Module layout

```
core/      state, topology, rules, apply(), legality, masks
encode/    state -> observation tensor
search/    ISMCTS, determinization
selfplay/  worker pool, game loop, trajectory writer
eval/      Elo harness, tournament, baselines
infer/     TorchScript or ONNX runtime wrapper
bind/      thin pybind11, tests and debugging only
```

`core` knows nothing about tensors or inference. `encode` depends on `core` alone. That separation is what lets the conformance tests and the human-playable debug mode run against `core` by itself.

`bind` is never in a training path. It exists so you can poke at a position in a notebook when a bot does something insane, and so you can play against Catanatron's bots without reimplementing their interface.

## State representation

Immutable topology is built once at startup and shared. It never enters the search tree.

Board constants: 19 hexes, 54 vertices, 72 edges, 9 ports.

```cpp
struct PublicState {
    std::array<uint8_t, NUM_VERTICES> vertex_owner;   // 2 bits owner + 1 bit city
    std::array<std::array<uint64_t, 2>, NUM_PLAYERS> roads;  // 72-bit set per player
    std::array<uint8_t, NUM_RES> bank;
    std::array<uint8_t, NUM_PLAYERS> hand_size;       // count only
    std::array<uint8_t, NUM_PLAYERS> dev_count;
    std::array<uint8_t, NUM_PLAYERS> knights_played;
    std::array<uint8_t, NUM_PLAYERS> public_vp;
    uint8_t dev_deck_remaining;
    uint8_t robber_hex;
    uint8_t current_player;
    Phase   phase;
    uint8_t dice;
    uint8_t longest_road_holder;
    uint8_t largest_army_holder;
    uint8_t longest_road_len;
    uint8_t trade_target;
    uint8_t trades_this_turn;
    std::array<int8_t, NUM_RES> open_offer;
    // plus piece counts remaining per player
};

struct HiddenState {
    std::array<std::array<uint8_t, NUM_RES>, NUM_PLAYERS> hands;
    std::array<std::array<uint8_t, NUM_DEV>, NUM_PLAYERS> dev_held;
    std::array<uint8_t, NUM_DEV> dev_deck;
    std::array<uint8_t, NUM_PLAYERS> dev_bought_this_turn;
};

struct GameState { PublicState pub; HiddenState hid; };
static_assert(std::is_trivially_copyable_v<GameState>);
```

Roughly 250 bytes, a few cache lines.

Design notes worth keeping:

- Roads are bitsets because longest-road is the hot computation and bit-parallel neighbor expansion is much faster. Vertices are bytes because owner plus settlement/city does not pack usefully.
- `hand_size` public, `hands` hidden. Mirrors reality: everyone can count your cards, nobody sees them.
- `trades_this_turn` is required even with trading off. Without a cap, random self-play produces infinite trade loops once trading is enabled and games never terminate. Cap at 6 to 8.
- `dev_bought_this_turn` is easy to forget and forgetting it is a real rules bug that lets a bot buy and immediately play a knight.

`resample_hidden(seed, observer)` is where correctness gets subtle. A naive resample hands someone cards that do not exist (bank plus all hands must equal 19 per resource) or contradicts what the observer has seen. Get this function and its invariant test right before building anything on top of it.

## The central abstraction: decision points

The engine does not expose "play a turn." It exposes:

```cpp
struct Decision {
    uint8_t player;   // not always current_player
    Phase phase;
    ActionMask mask;
};

Decision next_decision(const GameState&);
void apply(GameState&, Action);
```

`player` is separate from `current_player` because discards after a 7 belong to other players. Getting this wrong early means a rewrite, so build it in now.

Everything runs the same loop:

```cpp
while (!terminal(s)) {
    Decision d = next_decision(s);
    Action a = policy(s, d);
    apply(s, a);
}
```

Search, self-play, eval, and human play all use it unchanged.

## Action space

Factored heads, not a flat vector. Flat kills generalization on composite actions: with a flat space, Year of Plenty's 15 resource combinations are each learned independently, with no mechanism for feedback on one to inform the other 14.

Type head, 13 declared, 11 reachable:

```
ROLL, BUILD_ROAD, BUILD_SETTLEMENT, BUILD_CITY,
BUY_DEV, PLAY_DEV, BANK_EXCHANGE, MOVE_ROBBER,
STEAL, DISCARD, END_TURN
PROPOSE_TRADE    permanently masked
RESPOND_TRADE    permanently masked
```

Parameter heads:

| Head | Size | Used by |
| --- | --- | --- |
| vertex | 54 | settlement, city, setup settlement |
| edge | 72 | road, setup road, road building |
| tile | 19 | robber |
| devcard | 5 | which card to play |
| player | 3 | steal target, clockwise relative |
| res_give | 5 | bank exchange, monopoly |
| res_recv | 5 | bank exchange, year of plenty |

Log probability of a composite action is the sum over the heads the chosen type uses. Unused heads contribute zero. PPO is unchanged.

Year of Plenty uses both resource heads. Road Building is two sequential decision points rather than a second edge head, because the second placement depends on the first.

## Masking

Masks are computed in `core` alongside legality. Never in Python, never in the encoder. The mask is part of what `next_decision` returns.

Two levels: the type mask says which action types are legal at all, and per-head masks are conditional on the chosen type. `BUILD_SETTLEMENT` and `BUILD_CITY` share the vertex head with different masks, so the head takes the type as input.

Applied as `logits + log(mask)`: valid stays at 0, invalid goes to negative infinity, softmax zeroes them.

Invariant to test: the type mask is never all-zero. If it is, there is a rules bug, usually a phase that cannot terminate.

## Observation encoding

Written directly into a preallocated buffer. Three blocks.

**Tiles.** 19 entries, 61 dims each. One-hot token value, one-hot resource, robber flag, plus the six attached corners with building type and owner. Owner is relative to the deciding player (self, CW+1, CW+2, CW+3) so the network gets seat invariance for free. One-hot the token because 6 and 8 are best and 2 and 12 worst, so there is no useful ordering.

**Self.** Hand counts bucketed (1, 2, 3, 4, 5, 6-8, 8+), production vector per resource per dice value with cities counting double, ports held, VP one-hot, longest road and largest army flags and lengths, bank contents.

**Opponents.** Min and max possible holdings per resource given public information. Exact interval bounds, computed in `core` and updated inside `apply()`, not derived at encode time. This is the phase-1 belief layer and the baseline any learned model has to beat.

Dev cards go through an embedding, then attention, then a sum over the list, which handles the variable count cleanly.

## Self-play data flow

```
C++ worker                        Python trainer
----------                        --------------
N games stepping                  reads trajectory shards
     |                                   |
     v                                   v
batch observations --> TorchScript inference (in C++)
     |                                   |
     v                                   |
trajectory shards (disk) ------------->  train
     |                                   |
     <----------- new .pt checkpoint ----
```

The boundary is a file, not a function call. No pybind11 anywhere in the hot path. Workers poll for a new checkpoint between games and hot-swap.

Batched inference across games is the key performance detail. Each worker holds M games, steps them all to their next decision point, and batches M observations into one inference call. Games finish at different times, so you need a refill mechanism that keeps the batch full rather than draining it.

This is also why semi-asynchronous PPO is unnecessary here. The reference implementation went semi-async because each Python process held four separate policy networks and could not batch on GPU. One C++ inference engine batches across all games and all seats at once, including opponent policies, which are just different weight sets.

## Opponent sampling

One seat is the training policy. The other three are sampled from a checkpoint pool biased toward recent. This is required, not optional: pure latest-versus-latest self-play in a 4-player game overfits fast.

Checkpoint the opponent pool sampling distribution alongside the weights. The reference project lost significant progress to a crash because only the weights were saved and the pool had to be reconstructed ad hoc.

## Evaluation

Build this before the agent, not after.

- Fixed board seeds, full seat rotation, thousands of games per matchup. Catan variance will otherwise produce confident nonsense.
- Report Elo with confidence intervals. Always draw the 25% baseline on any win-rate plot.
- Every 100 updates, 1,200 games against three copies of the policy from 125 updates earlier, and separately from 250 updates earlier. The 250-lag curve sitting above the 125-lag curve is the sanity check that learning is real.
- Log average decisions per game. Declining game length is a cheap proxy for improvement.

Baselines in order: random-legal, greedy-heuristic (build what you can afford, prefer cities), then Catanatron's bots through `bind`.

## Build order

1. `core` with full rules including trading, trading flag off. Property tests: resource conservation (bank plus hands equals 19 each), VP monotonic, type mask never empty, longest road matches a brute-force reference on small positions.
2. `bind` plus a text or pygame debug view. You need to watch games to believe the rules.
3. `eval` with random and heuristic baselines.
4. `encode` plus `infer`, verified by getting sane outputs from an untrained net.
5. `selfplay` plus PPO. First target: beat the heuristic.
6. ISMCTS on top at eval time only.

Steps 1 to 3 are roughly 60% of the work and none of it is glamorous. Steps 1 to 3 plus a working PPO agent is already a complete project; the rest is upside.

## Open decisions

- Reward: win/loss plus VP gain only, or port the full shaping set and ablate down. Leaning sparse.
- Value head outputs 4 values (one per seat) even though phase 1 reads one entry, because max^n backup in search needs all four.
- Whether population-based training is in scope. It multiplies compute by N and is better framed as a diversity ablation than a strength maximizer.