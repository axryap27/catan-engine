"""The policy's move distribution: choose a move type, then that type's argument.

A move is (type, arg). The network outputs one logit per entry of the legal-move mask: 225 =
13 types followed by 212 argument slots, in encode.h's mask layout (ce.MASK_SECTIONS). A move's
probability is a product of two choices:

    P(type, arg) = softmax(type logits)[type] * softmax(logits of the type's section)[arg]

Each type owns at most one argument section (param_head in core/include/catan/action.h):

    BUILD_SETTLEMENT -> settlement    BUILD_CITY -> city       BUILD_ROAD -> edge
    MOVE_ROBBER -> tile               STEAL -> player          BANK_EXCHANGE, DISCARD -> resource
    PLAY_DEV -> devcard, except in the YOP_PICK and MONOPOLY_PICK phases, where its argument
                names a resource -> resource
    ROLL, BUY_DEV, END_TURN, PROPOSE_TRADE, RESPOND_TRADE -> none: arg is 0, P(type) alone

So the section depends on the phase, which is read from the observation (phase_of). The engine
guarantees that every row has a legal type, and that every legal type with a section has a
legal entry in it.

Illegal entries get NEG, a large finite negative, rather than -inf: their probability is still
exactly 0 in float32, but the entropy stays finite (with -inf it computes 0 * -inf = NaN).

Shapes: B is the batch (B = 0 is allowed). logits (B, MASK_SIZE) float; mask (B, MASK_SIZE)
uint8 or bool, nonzero = legal; phases (B,) int64 Phase values; types and args (B,) int64. All
functions are pure torch, work on any device, are differentiable in `logits`, and never modify
their inputs.
"""

import functools

import torch
import torch.nn.functional as F

from catan_rl.engine import ce

NEG = -1e9

NUM_TYPES = ce.NUM_ACTION_TYPES  # 13

# Argument sections, by section id. Id 0 means the type takes no argument.
SECTIONS: tuple[str, ...] = (
    "none", "settlement", "city", "edge", "tile", "devcard", "player", "resource")
MAX_SECTION = 72  # entries in the largest section (edge)


def _build_type_section() -> torch.Tensor:
    section_of = {
        ce.ActionType.BUILD_SETTLEMENT: "settlement",
        ce.ActionType.BUILD_CITY: "city",
        ce.ActionType.BUILD_ROAD: "edge",
        ce.ActionType.MOVE_ROBBER: "tile",
        ce.ActionType.PLAY_DEV: "devcard",
        ce.ActionType.STEAL: "player",
        ce.ActionType.BANK_EXCHANGE: "resource",
        ce.ActionType.DISCARD: "resource",
    }
    table = torch.zeros(NUM_TYPES, dtype=torch.int64)  # types not listed take no argument: 0
    for action_type, name in section_of.items():
        table[int(action_type)] = SECTIONS.index(name)
    return table


def _build_section_columns() -> torch.Tensor:
    """(len(SECTIONS), MAX_SECTION): entry [s, j] is the column of the 225-wide logits that holds
    argument j of section s. Entries past a section's size, and the whole "none" row, point at
    column MASK_SIZE: one extra column that _with_pad_column appends, holding NEG. This lets one
    gather pull "the argument row of each sample's type" out of the logits as a rectangle, even
    though the sections have different sizes."""
    table = torch.full((len(SECTIONS), MAX_SECTION), ce.MASK_SIZE, dtype=torch.int64)
    for s, name in enumerate(SECTIONS):
        if name != "none":
            start, size = ce.MASK_SECTIONS[name]
            table[s, :size] = torch.arange(start, start + size)
    return table


# Section id of each ActionType's argument outside YOP_PICK and MONOPOLY_PICK, shape (13,).
TYPE_SECTION: torch.Tensor = _build_type_section()
_SECTION_COLUMNS = _build_section_columns()

_RESOURCE = SECTIONS.index("resource")
_PLAY_DEV = int(ce.ActionType.PLAY_DEV)
_YOP_PICK = int(ce.Phase.YOP_PICK)
_MONOPOLY_PICK = int(ce.Phase.MONOPOLY_PICK)

# The Phase one-hot is the first len(Phase) = 11 bytes of the observation's global block
# (GLOBAL_PHASE = 0 in encode.h).
_GLOBAL_PHASE = 0
_PHASE_START = ce.OBS_SECTIONS["global"][0] + _GLOBAL_PHASE
_PHASE_END = _PHASE_START + len(ce.Phase)


@functools.lru_cache(maxsize=None)
def _tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """TYPE_SECTION and _SECTION_COLUMNS on `device` (a no-op on the CPU, one copy per device
    otherwise)."""
    return TYPE_SECTION.to(device), _SECTION_COLUMNS.to(device)


def phase_of(obs: torch.Tensor) -> torch.Tensor:
    """(B, OBS_SIZE) observations of any dtype -> (B,) int64 Phase values, read from the one-hot
    phase bytes at the start of the global block (GLOBAL_PHASE in encode.h)."""
    # A one-hot vector holds a single 1, so its argmax is the index of that 1: the phase.
    return obs[:, _PHASE_START:_PHASE_END].argmax(dim=1)


def arg_sections(types: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    """(B,) int64 section id of each type's argument in its phase (0 = no argument)."""
    # Written with elementwise ops only, so it accepts any broadcastable shapes: log_prob_entropy
    # passes (B, 13) types against (B, 1) phases to get every type's section at once.
    sections = _tables(types.device)[0][types]
    # In the two "pick a resource" phases the only legal type is PLAY_DEV, but its argument is a
    # resource, not a dev card.
    picking_resource = (phases == _YOP_PICK) | (phases == _MONOPOLY_PICK)
    return torch.where(picking_resource & (types == _PLAY_DEV), _RESOURCE, sections)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """logits with every illegal entry replaced by NEG."""
    # masked_fill is out of place (the caller's logits are untouched) and passes gradient through
    # the legal entries only.
    return logits.masked_fill(mask == 0, NEG)


def _with_pad_column(masked: torch.Tensor) -> torch.Tensor:
    """(B, MASK_SIZE) -> (B, MASK_SIZE + 1): a constant NEG column on the right, the target of
    the padding entries in _SECTION_COLUMNS."""
    return torch.cat([masked, masked.new_full((masked.shape[0], 1), NEG)], dim=1)


def _arg_logits(masked: torch.Tensor, sections: torch.Tensor) -> torch.Tensor:
    """(B, MAX_SECTION): row b is the masked logits of section sections[b], padded with NEG."""
    columns = _tables(masked.device)[1][sections]  # (B, MAX_SECTION) column numbers
    return _with_pad_column(masked).gather(1, columns)


def _log_prob(type_logp: torch.Tensor, arg_logp: torch.Tensor, types: torch.Tensor,
              sections: torch.Tensor, args: torch.Tensor) -> torch.Tensor:
    """log P(type, arg) = log P(type) + log P(arg | type), given both log-softmaxes."""
    # The two-part log-prob: a move is two independent choices, so its probability is a product
    # and its log-probability a sum. gather picks the chosen entry out of each row.
    type_part = type_logp.gather(1, types[:, None]).squeeze(1)
    arg_part = arg_logp.gather(1, args[:, None]).squeeze(1)
    # A type with no argument makes no second choice (probability 1, log 0). Its padding-only
    # row would otherwise contribute a meaningless -log(72) here.
    return type_part + arg_part.masked_fill(sections == 0, 0.0)


def sample(logits: torch.Tensor, mask: torch.Tensor, phases: torch.Tensor,
           generator: torch.Generator | None = None
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draws one legal move per row: (types, args, log_prob).

    log_prob = log P(type, arg), with its gradient to `logits`. arg is 0 for a type with no
    argument. `generator` (on logits' device) makes the draws reproducible; None uses torch's
    global RNG."""
    masked = masked_logits(logits, mask)
    type_logp = F.log_softmax(masked[:, :NUM_TYPES], dim=-1)  # the type logits are columns 0..12
    # Drawing is not differentiable, so the probabilities are detached; the gradient reaches
    # `logits` only through the log-probs that _log_prob gathers below.
    types = torch.multinomial(type_logp.detach().exp(), 1, generator=generator).squeeze(1)

    # Second choice: the argument, from the section of the type just drawn.
    sections = arg_sections(types, phases)
    arg_logp = F.log_softmax(_arg_logits(masked, sections), dim=-1)
    args = torch.multinomial(arg_logp.detach().exp(), 1, generator=generator).squeeze(1)
    args = args.masked_fill(sections == 0, 0)  # a type with no argument drew from pure padding
    return types, args, _log_prob(type_logp, arg_logp, types, sections, args)


def mode(logits: torch.Tensor, mask: torch.Tensor,
         phases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(types, args): the most likely legal type, then that type's most likely legal argument
    (play without sampling). arg is 0 for a type with no argument."""
    masked = masked_logits(logits, mask)
    types = masked[:, :NUM_TYPES].argmax(dim=-1)  # illegal types sit at NEG, so never win
    sections = arg_sections(types, phases)
    args = _arg_logits(masked, sections).argmax(dim=-1).masked_fill(sections == 0, 0)
    return types, args


def log_prob_entropy(logits: torch.Tensor, mask: torch.Tensor, phases: torch.Tensor,
                     types: torch.Tensor, args: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """(log_prob, entropy), both (B,), with gradients to `logits`.

    log_prob = log P(type, arg) of the given legal moves; for moves sample() drew it equals
    sample's log_prob. entropy is the exact entropy of each row's whole move distribution, not
    an estimate: H(type) + sum over types t of P(t) * H(argument of t), where a type with no
    argument adds 0."""
    masked = masked_logits(logits, mask)
    type_logp = F.log_softmax(masked[:, :NUM_TYPES], dim=-1)
    sections = arg_sections(types, phases)
    arg_logp = F.log_softmax(_arg_logits(masked, sections), dim=-1)
    log_prob = _log_prob(type_logp, arg_logp, types, sections, args)

    # Entropy of the two-step distribution, by the chain rule: H(type) + E_type[H(arg | type)].
    # Every term is p * log p. An illegal entry has p = 0 exactly and log p = about -1e9, which
    # is finite, so its term is a clean 0 (with -inf it would be 0 * -inf = NaN).
    type_p = type_logp.exp()
    type_entropy = -(type_p * type_logp).sum(dim=-1)

    # H(argument | type) for each of the 13 types: compute the entropy of each real section
    # once, (B, 7), then look up each type's section. Section 0 ("none") gets entropy 0.
    columns = _tables(masked.device)[1]
    section_logp = F.log_softmax(_with_pad_column(masked)[:, columns[1:]], dim=-1)  # (B, 7, 72)
    section_entropy = F.pad(-(section_logp.exp() * section_logp).sum(dim=-1), (1, 0))  # (B, 8)
    all_types = torch.arange(NUM_TYPES, device=types.device).expand(len(types), NUM_TYPES)
    type_sections = arg_sections(all_types, phases[:, None])  # (B, 13): phase-dependent
    arg_entropy = section_entropy.gather(1, type_sections)
    # A section with no legal entry (say, cities when none can be built) comes out as a uniform
    # distribution here, but only for a type whose P(type) is exactly 0, so it adds nothing.
    entropy = type_entropy + (type_p * arg_entropy).sum(dim=-1)
    return log_prob, entropy
