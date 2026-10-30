"""Python bindings for the C++ Catan engine.

Built by CMake with -DCATAN_BUILD_PYTHON=ON (preset `python`) as the extension module
`catan_engine`. For debugging, notebooks, tests and Python-driven training; the rules, legal
moves, belief tracking and encoding all run in C++.

Conventions
- Players are absolute seats 0..3. "No player" is None in scalars and -1 in numpy arrays.
- An action is a pair (type, arg) of ints: an ActionType and its parameter (vertex, edge, hex,
  dev card, resource, or for STEAL the victim relative to the decider: 0 = next seat,
  1 = across, 2 = previous). See core/include/catan/action.h.
- Observations and masks are uint8 numpy arrays laid out as in encode/include/catan/encode.h.
  Every returned array is a fresh copy, safe to keep (e.g. in a rollout buffer).
- Bad input (illegal action, wrong decision kind, out-of-range seat or index, wrong array
  shape or dtype) raises ValueError.
- Everything is deterministic in the seed: the board is random_board(seed) (or the standard
  board), and dice, dev draws and steals are drawn from the seed as the C++ drivers do.
"""

import enum

import numpy as np
import numpy.typing as npt

ENCODE_VERSION: int
OBS_SIZE: int  # 2355
MASK_SIZE: int  # 225
NUM_PLAYERS: int  # 4
NUM_ACTION_TYPES: int  # 13
VICTORY_POINTS_TO_WIN: int  # 10
TURN_CAP: int  # 1000

# Observation sections: name -> (byte offset, rows, bytes per row); see OBS_SECTIONS in encode.h.
OBS_SECTIONS: dict[str, tuple[int, int, int]]
# Mask sections: "type", "settlement", "city", "edge", "tile", "devcard", "player",
# "resource" -> (byte offset, size); see MASK_* in encode.h.
MASK_SECTIONS: dict[str, tuple[int, int]]

class ActionType(enum.IntEnum):
    ROLL = 0
    BUILD_ROAD = 1
    BUILD_SETTLEMENT = 2
    BUILD_CITY = 3
    BUY_DEV = 4
    PLAY_DEV = 5
    BANK_EXCHANGE = 6
    MOVE_ROBBER = 7
    STEAL = 8
    DISCARD = 9
    END_TURN = 10
    PROPOSE_TRADE = 11
    RESPOND_TRADE = 12

class Phase(enum.IntEnum):
    SETUP = 0
    ROLL = 1
    DISCARD = 2
    MOVE_ROBBER = 3
    STEAL = 4
    MAIN = 5
    ROAD_BUILDING = 6
    YOP_PICK = 7
    MONOPOLY_PICK = 8
    EXCHANGE_RECV = 9
    TERMINAL = 10

class Resource(enum.IntEnum):
    BRICK = 0
    LUMBER = 1
    WOOL = 2
    GRAIN = 3
    ORE = 4

class DevCard(enum.IntEnum):
    KNIGHT = 0
    VICTORY_POINT = 1
    ROAD_BUILDING = 2
    YEAR_OF_PLENTY = 3
    MONOPOLY = 4

class DecisionKind(enum.IntEnum):
    PLAYER = 0
    CHANCE = 1
    TERMINAL = 2

class EventType(enum.IntEnum):
    SETTLEMENT_BUILT = 0
    CITY_BUILT = 1
    ROAD_BUILT = 2
    GAIN = 3
    SPEND = 4
    DISCARD = 5
    DICE_ROLLED = 6
    ROBBER_MOVED = 7
    STEAL = 8
    DEV_BOUGHT = 9
    DEV_PLAYED = 10
    MONOPOLY = 11
    LONGEST_ROAD = 12
    LARGEST_ARMY = 13
    TURN_ENDED = 14
    GAME_OVER = 15

class Event:
    """One thing that happened, carrying the full truth (see core/include/catan/event.h)."""

    type: EventType
    player: int | None
    target: int | None
    a: int
    b: int
    def describe(self, observer: int | None = None) -> str:
        """describe_event: hides a steal's resource / a bought card from other observers."""
    def __str__(self) -> str: ...  # describe(None)
    def __repr__(self) -> str: ...

def random_below(seed: int, stream: int, index: int, bound: int) -> int:
    """The engine's counter-based RNG (rng.h): uniform in [0, bound), pure in its inputs."""

def topology() -> dict[str, list]:
    """Fixed board geometry from topology.h: "hex_coord" [(q, r)] * 19, "hex_vertices",
    "hex_edges", "hex_neighbors" (None for sea) [[6 ints]] * 19, "vertex_hexes",
    "vertex_vertices", "vertex_edges" (variable-length lists, padding dropped) * 54,
    "edge_vertices" [(v0, v1)] * 72, "port_slots" [(edge, v0, v1, hex)] * 9."""

class Game:
    """One game: board, full state, every observer's belief, and the seed for chance.

    With auto_chance (default), chance decisions (dice, dev draws, steals) are resolved from
    the seed right after each step, so a game only ever waits on a player or is over."""

    def __init__(self, seed: int, standard_board: bool = False,
                 auto_chance: bool = True) -> None: ...
    @property
    def seed(self) -> int: ...

    # ---- state (read-only) ----
    @property
    def kind(self) -> DecisionKind: ...
    @property
    def phase(self) -> Phase: ...
    @property
    def decider(self) -> int | None:
        """Seat that must act now (not always current_player: discards); None unless PLAYER."""
    @property
    def current_player(self) -> int: ...
    @property
    def turn(self) -> int: ...
    @property
    def ply(self) -> int: ...
    @property
    def done(self) -> bool: ...  # TERMINAL
    @property
    def winner(self) -> int | None:
        """The seat that reached 10 VP; None while playing or if the turn cap ended the game."""
    @property
    def public_vp(self) -> list[int]: ...
    @property
    def vp(self) -> list[int]:
        """True VP: public VP plus hidden VP cards. Full truth; never show it to a policy."""
    def hand(self, player: int) -> list[int]:
        """True resource counts per Resource. Full truth (debugging only)."""
    def dev_cards(self, player: int) -> list[int]:
        """True dev cards held per DevCard. Full truth (debugging only)."""
    def board(self) -> dict[str, list]:
        """"hex_type" [int HexType] * 19 (5 = desert), "token" [int] * 19 (0 = desert),
        "port_type" [int PortType] * 9 (5 = 3:1) per port slot, "robber" int hex."""

    # ---- acting ----
    def legal_actions(self) -> list[tuple[int, int]]:
        """Every legal (type, arg), in the engine's canonical order (nth_legal); [] unless
        kind is PLAYER."""
    def is_legal(self, type: int, arg: int = 0) -> bool: ...
    def step(self, type: int, arg: int = 0) -> list[Event]:
        """Apply the decider's action, then (auto_chance) resolve chance until a player must
        act or the game ends. Returns every event, in order. Raises ValueError, leaving the
        game unchanged, if kind is not PLAYER or the action is illegal."""
    def chance_weights(self) -> list[int]:
        """Outcome weights of the pending chance decision, one per outcome (11 for dice, 5 for
        dev draws and steals; index = outcome, see decision.h); [] unless kind is CHANCE (only
        reachable with auto_chance=False)."""
    def apply_chance(self, outcome: int) -> list[Event]:
        """Resolve the pending chance decision; ValueError if none or weight 0."""
    def greedy_action(self) -> tuple[int, int]:
        """The greedy baseline's choice for the decider, with the rng value match.cpp uses
        (random_u64(seed, 100 + decider, ply)). ValueError unless kind is PLAYER."""

    # ---- network input ----
    def observation(self, observer: int | None = None) -> npt.NDArray[np.uint8]:
        """encode() for `observer` (default: the decider), shape (OBS_SIZE,). Uses only what
        that seat may know. ValueError if observer is None and no player is to act."""
    def mask(self) -> npt.NDArray[np.uint8]:
        """encode_mask() of the current decision, shape (MASK_SIZE,); all zero unless PLAYER."""

    # ---- text ----
    def describe(self, type: int, arg: int = 0) -> str:
        """describe_action for the current decider and phase."""
    def render(self, observer: int | None = None, vertex_ids: bool = False,
               edge_ids: bool = False) -> str:
        """render_state; observer None is the god view."""
    def __str__(self) -> str: ...  # render()

    # ---- misc ----
    def copy(self) -> "Game":
        """Independent deep copy: board, state, belief, seed and auto_chance."""
    def state_hash(self) -> int: ...  # state_hash(), as pinned by the C++ golden tests

class VecGames:
    """n independent games for batched play, e.g. PPO rollouts.

    Every game always waits on a PLAYER decision: chance is resolved from the seed, and a game
    that ends is reported by step() and restarted at once on the next unused seed. Seeds are
    handed out in order: game i starts on first_seed + i, and restarts take first_seed + n,
    first_seed + n + 1, ... in the order they happen (within one step, by game index), so a
    run is reproducible from (n, first_seed) and the actions. Each game resolves chance and
    greedy_actions exactly as a Game with that seed would. The GIL is released while stepping
    and encoding."""

    def __init__(self, n: int, first_seed: int = 1, standard_board: bool = False) -> None: ...
    @property
    def n(self) -> int: ...
    @property
    def seeds(self) -> npt.NDArray[np.uint64]:
        """Current seed of each game, shape (n,)."""

    def observe(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.int8]]:
        """(obs (n, OBS_SIZE), mask (n, MASK_SIZE), decider (n,)): each game's current decider,
        its observation and the legal-move mask."""
    def step(self, types: npt.ArrayLike, args: npt.ArrayLike) -> tuple[
            npt.NDArray[np.bool_], npt.NDArray[np.int8], npt.NDArray[np.uint8],
            npt.NDArray[np.uint16]]:
        """Apply one action per game (types and args: integer arrays of shape (n,)).

        Returns (done (n,), winner (n,), vp (n, 4), turns (n,)). For a game that ended in this
        step: done is True, winner its winning seat or -1 if the turn cap ended it, vp the true
        final VP of each seat, turns its turn count; that game has already restarted. For the
        others: False, -1, zeros, 0. Raises ValueError naming the first illegal game, before
        any game is changed."""
    def greedy_actions(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """(types (n,), args (n,)): the greedy baseline's choice for every game's decider."""
    def game(self, i: int) -> Game:
        """A copy of game i as a Game (auto_chance on), for inspection or rendering."""
