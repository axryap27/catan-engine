"""Tests for catan_rl.actions, written from its contract (the module docstring), not its code.

The reference distribution below is computed independently in numpy (float64) from the
contract's formula: P(type, arg) = softmax(legal type logits)[type] * softmax(legal logits of
the type's section)[arg], with the section of each type from the contract's table.
"""

import math

import numpy as np
import pytest
import torch

from catan_rl import actions
from catan_rl.engine import ce

A = ce.ActionType
PH = ce.Phase
NUM_TYPES = ce.NUM_ACTION_TYPES
MASK_SIZE = ce.MASK_SIZE

# The contract's argument section of each type outside the YOP_PICK and MONOPOLY_PICK phases.
CONTRACT_SECTION = {
    A.ROLL: "none",
    A.BUILD_ROAD: "edge",
    A.BUILD_SETTLEMENT: "settlement",
    A.BUILD_CITY: "city",
    A.BUY_DEV: "none",
    A.PLAY_DEV: "devcard",
    A.BANK_EXCHANGE: "resource",
    A.MOVE_ROBBER: "tile",
    A.STEAL: "player",
    A.DISCARD: "resource",
    A.END_TURN: "none",
    A.PROPOSE_TRADE: "none",
    A.RESPOND_TRADE: "none",
}
PICK_PHASES = (int(PH.YOP_PICK), int(PH.MONOPOLY_PICK))
NON_PICK_PHASES = [int(p) for p in PH if p not in (PH.YOP_PICK, PH.MONOPOLY_PICK, PH.TERMINAL)]
LOGIT_KINDS = ["zero", "normal", "wide", "huge_illegal"]


# ---- reference implementation (numpy, from the contract) -----------------------------------

def section_of(t: int, phase: int) -> str:
    if t == A.PLAY_DEV and phase in PICK_PHASES:
        return "resource"
    return CONTRACT_SECTION[A(t)]


def section_indices(name: str) -> np.ndarray:
    off, size = ce.MASK_SECTIONS[name]
    return np.arange(off, off + size)


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    return x - m - math.log(np.exp(x - m).sum())


def ref_log_probs(logits, mask, phase: int) -> dict[tuple[int, int], float]:
    """{(type, arg): log P(type, arg)} over every move the mask allows."""
    z = np.asarray(logits, dtype=np.float64)
    m = np.asarray(mask) != 0
    types = np.flatnonzero(m[:NUM_TYPES])
    out = {}
    for t, lpt in zip(types.tolist(), log_softmax(z[types])):
        name = section_of(t, phase)
        if name == "none":
            out[(t, 0)] = float(lpt)
            continue
        idx = section_indices(name)
        legal = np.flatnonzero(m[idx])
        for a, lpa in zip(legal.tolist(), log_softmax(z[idx[legal]])):
            out[(t, a)] = float(lpt + lpa)
    return out


def ref_entropy(dist: dict) -> float:
    lp = np.array(list(dist.values()))
    return float(-(np.exp(lp) * lp).sum())


def ref_mode(logits, mask, phase: int) -> tuple[int, int]:
    z = np.asarray(logits, dtype=np.float64)
    m = np.asarray(mask) != 0
    types = np.flatnonzero(m[:NUM_TYPES])
    t = int(types[np.argmax(z[types])])
    name = section_of(t, phase)
    if name == "none":
        return t, 0
    idx = section_indices(name)
    legal = np.flatnonzero(m[idx])
    return t, int(legal[np.argmax(z[idx[legal]])])


def ref_grad_log_prob(logits, mask, phase: int, t: int, a: int) -> np.ndarray:
    """d log P(t, a) / d logits: onehot - softmax on the type head and on t's section only."""
    z = np.asarray(logits, dtype=np.float64)
    m = np.asarray(mask) != 0
    g = np.zeros(MASK_SIZE)
    types = np.flatnonzero(m[:NUM_TYPES])
    g[types] -= np.exp(log_softmax(z[types]))
    g[t] += 1.0
    name = section_of(t, phase)
    if name != "none":
        idx = section_indices(name)
        legal = idx[m[idx]]
        g[legal] -= np.exp(log_softmax(z[legal]))
        g[idx[a]] += 1.0
    return g


def ref_grad_entropy(logits, mask, phase: int) -> np.ndarray:
    """d/d logits of H(type) + sum_t P(t) H(arg | t), derived by hand."""
    z = np.asarray(logits, dtype=np.float64)
    m = np.asarray(mask) != 0
    g = np.zeros(MASK_SIZE)
    types = np.flatnonzero(m[:NUM_TYPES])
    lp = log_softmax(z[types])
    p = np.exp(lp)
    h_type = -(p * lp).sum()
    h_arg = np.zeros(len(types))
    for k, t in enumerate(types.tolist()):
        name = section_of(t, phase)
        if name == "none":
            continue
        idx = section_indices(name)
        legal = idx[m[idx]]
        lq = log_softmax(z[legal])
        q = np.exp(lq)
        h_arg[k] = -(q * lq).sum()
        g[legal] += p[k] * (-q * (lq + h_arg[k]))
    e = (p * h_arg).sum()
    g[types] += -p * (lp + h_type) + p * (h_arg - e)
    return g


# ---- helpers ----------------------------------------------------------------------------------

def tensors(positions, idx=None):
    """(obs uint8, mask uint8, phases int64) torch tensors, all rows or rows `idx`."""
    sel = slice(None) if idx is None else np.asarray(idx)
    return (torch.from_numpy(positions.obs[sel].copy()),
            torch.from_numpy(positions.mask[sel].copy()),
            torch.from_numpy(positions.phases[sel].copy()))


def make_logits(kind: str, mask: torch.Tensor, seed: int) -> torch.Tensor:
    n = mask.shape[0]
    if kind == "zero":
        return torch.zeros(n, MASK_SIZE)
    z = torch.randn(n, MASK_SIZE, generator=torch.Generator().manual_seed(seed))
    if kind == "normal":
        return 2.0 * z
    if kind == "wide":
        return 10.0 * z
    if kind == "huge_illegal":  # the illegal entries would win every softmax if not masked
        return torch.where(mask != 0, z, torch.full_like(z, 1e6))
    raise AssertionError(kind)


def all_legal_moves(positions):
    """Flattened (rows, types, args) over every position's legal_actions()."""
    rows, types, args = [], [], []
    for i, moves in enumerate(positions.legal):
        for t, a in moves:
            rows.append(i)
            types.append(t)
            args.append(a)
    return (torch.tensor(rows, dtype=torch.int64), torch.tensor(types, dtype=torch.int64),
            torch.tensor(args, dtype=torch.int64))


def assert_moves_legal(positions, idx, types, args):
    assert types.dtype == torch.int64 and args.dtype == torch.int64
    for i, t, a in zip(idx, types.tolist(), args.tolist()):
        phase = int(positions.phases[i])
        assert positions.games[i].is_legal(t, a), (
            f"position {i} ({PH(phase).name}): drew illegal move ({A(t).name}, {a})")
        if section_of(t, phase) == "none":
            assert a == 0, f"{A(t).name} takes no argument but got arg {a}"


def frequency_position(positions, which: str) -> int:
    """A fixed position for the sampling-frequency tests."""
    ph, legal = positions.phases, positions.legal

    def n_types(i):
        return len({t for t, _ in legal[i]})

    if which == "main_many_types":
        cands = positions.where(lambda i: ph[i] == PH.MAIN)
        return max(cands, key=lambda i: (n_types(i), len(legal[i]), -i))
    if which == "roll_with_knight":
        return positions.where(lambda i: ph[i] == PH.ROLL and n_types(i) >= 2)[0]
    phase = {"yop_pick": PH.YOP_PICK, "monopoly_pick": PH.MONOPOLY_PICK,
             "discard": PH.DISCARD}[which]
    return positions.where(lambda i: ph[i] == phase and len(legal[i]) >= 2)[0]


# ---- the fixture itself --------------------------------------------------------------------

def test_fixture_covers_every_nonterminal_phase(positions):
    """The positions must exercise every phase; YOP_PICK and MONOPOLY_PICK in particular are
    where PLAY_DEV's argument switches to the resource section."""
    counts = {p.name: int((positions.phases == int(p)).sum()) for p in PH}
    for p in PH:
        if p == PH.TERMINAL:
            assert counts[p.name] == 0
        else:
            assert counts[p.name] >= 10, counts
    for i, g in enumerate(positions.games):
        assert int(g.phase) == positions.phases[i] and not g.done
    assert positions.seconds < 5.0


def test_fixture_has_interesting_type_mixes(positions):
    """ROLL with a knight playable (PLAY_DEV -> devcard outside the pick phases) and MAIN with
    several types legal at once (several sections live in one row)."""
    ph, legal = positions.phases, positions.legal
    types = [{t for t, _ in moves} for moves in legal]
    assert any(ph[i] == PH.ROLL and A.PLAY_DEV in types[i] for i in range(len(ph)))
    assert any(ph[i] == PH.MAIN and A.PLAY_DEV in types[i] for i in range(len(ph)))
    assert any(ph[i] == PH.MAIN and A.BANK_EXCHANGE in types[i] for i in range(len(ph)))
    assert max(len(types[i]) for i in range(len(ph)) if ph[i] == PH.MAIN) >= 4


def test_contract_mapping_reproduces_legal_actions(positions):
    """Reading the mask through the contract's type -> section table must give exactly the
    engine's legal_actions(); otherwise the reference distribution below is wrong."""
    for i in range(len(positions)):
        moves = set(ref_log_probs(np.zeros(MASK_SIZE), positions.mask[i], positions.phases[i]))
        assert moves == set(positions.legal[i]), PH(int(positions.phases[i])).name


# ---- constants -----------------------------------------------------------------------------

def test_constants():
    assert actions.SECTIONS == (
        "none", "settlement", "city", "edge", "tile", "devcard", "player", "resource")
    for name in actions.SECTIONS[1:]:
        assert name in ce.MASK_SECTIONS
    sizes = [size for name, (_, size) in ce.MASK_SECTIONS.items() if name != "type"]
    assert actions.MAX_SECTION == max(sizes) == 72
    assert math.isfinite(actions.NEG) and actions.NEG < -1e6
    # Masked entries must get probability exactly 0 in float32.
    p = torch.softmax(torch.tensor([0.0, actions.NEG], dtype=torch.float32), dim=0)
    assert p[1].item() == 0.0 and torch.isfinite(torch.tensor(actions.NEG, dtype=torch.float32))


def test_type_section_table():
    ts = torch.as_tensor(actions.TYPE_SECTION)
    assert ts.shape == (NUM_TYPES,)
    expected = [actions.SECTIONS.index(CONTRACT_SECTION[A(t)]) for t in range(NUM_TYPES)]
    assert ts.tolist() == expected


def test_arg_sections_matches_contract():
    """Every type in every ordinary phase, plus PLAY_DEV in the pick phases (-> resource)."""
    types, phases, expected = [], [], []
    for phase in NON_PICK_PHASES:
        for t in range(NUM_TYPES):
            types.append(t)
            phases.append(phase)
            expected.append(actions.SECTIONS.index(CONTRACT_SECTION[A(t)]))
    for phase in PICK_PHASES:
        types.append(int(A.PLAY_DEV))
        phases.append(phase)
        expected.append(actions.SECTIONS.index("resource"))
    out = actions.arg_sections(torch.tensor(types), torch.tensor(phases))
    assert out.dtype == torch.int64 and out.shape == (len(types),)
    assert out.tolist() == expected


# ---- phase_of and masked_logits -------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.uint8, torch.int64, torch.float32, torch.float64])
def test_phase_of_matches_game_phase(positions, dtype):
    obs, _, phases = tensors(positions)
    out = actions.phase_of(obs.to(dtype))
    assert out.dtype == torch.int64 and out.shape == phases.shape
    assert torch.equal(out, phases)


def test_phase_of_empty_batch():
    out = actions.phase_of(torch.zeros(0, ce.OBS_SIZE, dtype=torch.uint8))
    assert out.shape == (0,) and out.dtype == torch.int64


@pytest.mark.parametrize("mask_kind", ["uint8", "uint8_255", "bool"])
def test_masked_logits(positions, mask_kind):
    """Legal entries pass through unchanged, illegal ones become exactly NEG; any nonzero mask
    byte means legal."""
    _, mask, _ = tensors(positions)
    logits = make_logits("normal", mask, seed=1)
    m = {"uint8": mask, "uint8_255": mask * 255, "bool": mask != 0}[mask_kind]
    out = actions.masked_logits(logits, m)
    legal = mask != 0
    assert out.shape == logits.shape and out.dtype == logits.dtype
    assert torch.equal(out[legal], logits[legal])
    assert torch.all(out[~legal] == actions.NEG)


# ---- legality -------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", LOGIT_KINDS)
def test_sample_draws_only_legal_moves(positions, kind):
    """Any illegal draw means the mask or the type -> section mapping is applied wrongly."""
    _, mask, phases = tensors(positions)
    gen = torch.Generator().manual_seed(7)
    for rep in range(3):
        logits = make_logits(kind, mask, seed=100 + rep)
        types, args, log_prob = actions.sample(logits, mask, phases, generator=gen)
        assert types.shape == args.shape == log_prob.shape == (len(positions),)
        assert torch.isfinite(log_prob).all()
        assert_moves_legal(positions, range(len(positions)), types, args)


@pytest.mark.parametrize("kind", LOGIT_KINDS)
def test_mode_is_argmax_type_then_argmax_arg(positions, kind):
    _, mask, phases = tensors(positions)
    logits = make_logits(kind, mask, seed=3)
    types, args = actions.mode(logits, mask, phases)
    assert types.shape == args.shape == (len(positions),)
    assert_moves_legal(positions, range(len(positions)), types, args)
    if kind == "zero":
        return  # ties: any legal move is a valid mode
    for i in range(len(positions)):
        want = ref_mode(logits[i].numpy(), positions.mask[i], positions.phases[i])
        assert (types[i].item(), args[i].item()) == want, PH(int(positions.phases[i])).name


# ---- the distribution ------------------------------------------------------------------------

@pytest.mark.parametrize("kind", LOGIT_KINDS)
def test_probabilities_sum_to_one_over_legal_actions(positions, kind):
    """sum over game.legal_actions() of P(move) == 1: all mass on exactly the legal moves. A
    wrong section for any type (e.g. PLAY_DEV in YOP_PICK/MONOPOLY_PICK) breaks this."""
    _, mask, phases = tensors(positions)
    logits = make_logits(kind, mask, seed=11)
    rows, types, args = all_legal_moves(positions)
    log_prob, _ = actions.log_prob_entropy(logits[rows], mask[rows], phases[rows], types, args)
    assert torch.isfinite(log_prob).all()
    total = torch.zeros(len(positions), dtype=torch.float64)
    total.index_add_(0, rows, log_prob.double().exp())
    worst = int((total - 1).abs().argmax())
    assert abs(total[worst].item() - 1) < 1e-5, (
        f"position {worst} ({PH(int(positions.phases[worst])).name}): mass {total[worst]:.6f}")


@pytest.mark.parametrize("kind", LOGIT_KINDS)
def test_log_prob_and_entropy_match_reference(positions, kind):
    """log_prob equals the contract formula for every legal move; entropy equals
    -sum p log p over the legal moves (exact, not a sample estimate)."""
    _, mask, phases = tensors(positions)
    logits = make_logits(kind, mask, seed=12)
    rows, types, args = all_legal_moves(positions)
    log_prob, entropy = actions.log_prob_entropy(
        logits[rows], mask[rows], phases[rows], types, args)
    assert log_prob.shape == entropy.shape == rows.shape
    assert log_prob.dtype == entropy.dtype == torch.float32
    refs = [ref_log_probs(logits[i].numpy(), positions.mask[i], positions.phases[i])
            for i in range(len(positions))]
    want_lp = np.array([refs[r][(t, a)]
                        for r, t, a in zip(rows.tolist(), types.tolist(), args.tolist())])
    want_h = np.array([ref_entropy(refs[r]) for r in rows.tolist()])
    np.testing.assert_allclose(log_prob.double().numpy(), want_lp, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(entropy.double().numpy(), want_h, rtol=1e-5, atol=1e-5)


def test_single_move_rows_have_zero_entropy(positions):
    idx = positions.where(lambda i: len(positions.legal[i]) == 1)
    assert idx, "fixture should contain forced moves"
    _, mask, phases = tensors(positions, idx)
    logits = make_logits("normal", mask, seed=5)
    types = torch.tensor([positions.legal[i][0][0] for i in idx])
    args = torch.tensor([positions.legal[i][0][1] for i in idx])
    log_prob, entropy = actions.log_prob_entropy(logits, mask, phases, types, args)
    assert log_prob.abs().max().item() < 1e-5
    assert entropy.abs().max().item() < 1e-5


@pytest.mark.parametrize("kind", ["normal", "wide", "huge_illegal"])
def test_sample_log_prob_equals_log_prob_entropy(positions, kind):
    _, mask, phases = tensors(positions)
    logits = make_logits(kind, mask, seed=21)
    types, args, log_prob = actions.sample(
        logits, mask, phases, generator=torch.Generator().manual_seed(1))
    again, _ = actions.log_prob_entropy(logits, mask, phases, types, args)
    torch.testing.assert_close(log_prob, again, rtol=1e-5, atol=1e-5)
    want = [ref_log_probs(logits[i].numpy(), positions.mask[i], positions.phases[i])[(t, a)]
            for i, (t, a) in enumerate(zip(types.tolist(), args.tolist()))]
    np.testing.assert_allclose(log_prob.double().numpy(), want, rtol=1e-5, atol=1e-5)


def test_illegal_logits_do_not_matter(positions):
    """Changing only illegal logits leaves log_prob, entropy and mode unchanged."""
    _, mask, phases = tensors(positions)
    base = make_logits("normal", mask, seed=31)
    rows, types, args = all_legal_moves(positions)
    ref_lp, ref_h = actions.log_prob_entropy(base[rows], mask[rows], phases[rows], types, args)
    ref_mode_t, ref_mode_a = actions.mode(base, mask, phases)
    for fill in (1e6, -1e6, 0.0):
        other = torch.where(mask != 0, base, torch.full_like(base, fill))
        lp, h = actions.log_prob_entropy(other[rows], mask[rows], phases[rows], types, args)
        torch.testing.assert_close(lp, ref_lp, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(h, ref_h, rtol=1e-6, atol=1e-6)
        t, a = actions.mode(other, mask, phases)
        assert torch.equal(t, ref_mode_t) and torch.equal(a, ref_mode_a)


FREQ_DRAWS = 20_000


@pytest.mark.parametrize(
    "which", ["main_many_types", "roll_with_knight", "yop_pick", "monopoly_pick", "discard"])
def test_sample_frequencies_match_probabilities(positions, which):
    """Draw one row many times: each move's frequency (and each type's) must be within 5
    standard errors of the reference probability. Catches biased or mis-sectioned sampling
    even where log_prob_entropy is right."""
    i = frequency_position(positions, which)
    phase = int(positions.phases[i])
    _, mask, phases = tensors(positions, [i] * FREQ_DRAWS)
    row = torch.randn(MASK_SIZE, generator=torch.Generator().manual_seed(1000 + i))
    logits = row.expand(FREQ_DRAWS, MASK_SIZE).contiguous()
    types, args, _ = actions.sample(
        logits, mask, phases, generator=torch.Generator().manual_seed(4242))
    ref = ref_log_probs(row.numpy(), positions.mask[i], phase)
    assert len(ref) >= 2
    drawn = list(zip(types.tolist(), args.tolist()))
    counts: dict = {}
    for m in drawn:
        counts[m] = counts.get(m, 0) + 1
    assert set(counts) <= set(ref), f"illegal draws: {set(counts) - set(ref)}"

    def check(observed, p, what):
        se = math.sqrt(p * (1 - p) / FREQ_DRAWS)
        assert abs(observed / FREQ_DRAWS - p) <= 5 * se + 1e-9, (
            f"{PH(phase).name} {what}: freq {observed / FREQ_DRAWS:.4f} vs p {p:.4f}")

    for move, lp in ref.items():
        check(counts.get(move, 0), math.exp(lp), f"move {move}")
    type_p: dict = {}
    for (t, _), lp in ref.items():
        type_p[t] = type_p.get(t, 0.0) + math.exp(lp)
    for t, p in type_p.items():
        check(int((types == t).sum()), p, f"type {A(t).name}")


# ---- gradients --------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["normal", "huge_illegal"])
def test_log_prob_gradient_matches_reference(positions, kind):
    """d log P / d logits is (onehot - softmax) on the type head and on the chosen type's
    section only: zero on every other section (even legal entries) and on illegal entries."""
    _, mask, phases = tensors(positions)
    logits = make_logits(kind, mask, seed=41).requires_grad_(True)
    types, args, _ = actions.sample(
        logits.detach(), mask, phases, generator=torch.Generator().manual_seed(2))
    log_prob, _ = actions.log_prob_entropy(logits, mask, phases, types, args)
    (grad,) = torch.autograd.grad(log_prob.sum(), logits)
    assert torch.isfinite(grad).all()
    assert torch.all(grad[mask == 0] == 0), "illegal logits must get exactly zero gradient"
    for i in range(len(positions)):
        want = ref_grad_log_prob(logits[i].detach().numpy(), positions.mask[i],
                                 positions.phases[i], types[i].item(), args[i].item())
        np.testing.assert_allclose(grad[i].double().numpy(), want, rtol=1e-4, atol=1e-5,
                                   err_msg=PH(int(positions.phases[i])).name)


@pytest.mark.parametrize("kind", ["normal", "huge_illegal"])
def test_entropy_gradient_matches_reference(positions, kind):
    _, mask, phases = tensors(positions)
    logits = make_logits(kind, mask, seed=42).requires_grad_(True)
    types = torch.tensor([moves[0][0] for moves in positions.legal])
    args = torch.tensor([moves[0][1] for moves in positions.legal])
    _, entropy = actions.log_prob_entropy(logits, mask, phases, types, args)
    (grad,) = torch.autograd.grad(entropy.sum(), logits)
    assert torch.isfinite(grad).all()
    assert torch.all(grad[mask == 0] == 0), "illegal logits must get exactly zero gradient"
    for i in range(len(positions)):
        want = ref_grad_entropy(logits[i].detach().numpy(), positions.mask[i],
                                positions.phases[i])
        np.testing.assert_allclose(grad[i].double().numpy(), want, rtol=1e-4, atol=1e-5,
                                   err_msg=PH(int(positions.phases[i])).name)


def test_sample_log_prob_is_differentiable(positions):
    _, mask, phases = tensors(positions)
    logits = make_logits("normal", mask, seed=43).requires_grad_(True)
    types, args, log_prob = actions.sample(
        logits, mask, phases, generator=torch.Generator().manual_seed(3))
    assert log_prob.requires_grad
    (grad,) = torch.autograd.grad(log_prob.sum(), logits)
    assert torch.isfinite(grad).all() and torch.all(grad[mask == 0] == 0)
    for i in range(0, len(positions), 7):
        want = ref_grad_log_prob(logits[i].detach().numpy(), positions.mask[i],
                                 positions.phases[i], types[i].item(), args[i].item())
        np.testing.assert_allclose(grad[i].double().numpy(), want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("scale", [30.0, 300.0])
def test_saturated_logits_stay_finite(positions, scale):
    """Large logits (a confident trained net) push some legal probabilities below float32's
    range; log_prob, entropy and their gradients must still be finite (no log(0), 0 * inf)."""
    _, mask, phases = tensors(positions)
    z = torch.randn(len(positions), MASK_SIZE, generator=torch.Generator().manual_seed(9))
    logits = torch.where(mask != 0, scale * z, torch.full_like(z, 1e6)).requires_grad_(True)
    rows, types, args = all_legal_moves(positions)
    log_prob, entropy = actions.log_prob_entropy(
        logits[rows], mask[rows], phases[rows], types, args)
    assert torch.isfinite(log_prob).all() and torch.isfinite(entropy).all()
    assert (log_prob <= 1e-6).all() and (entropy >= -1e-5).all()
    (grad,) = torch.autograd.grad(log_prob.sum() + entropy.sum(), logits)
    assert torch.isfinite(grad).all()
    assert torch.all(grad[mask == 0] == 0)
    t, a, lp = actions.sample(logits, mask, phases, generator=torch.Generator().manual_seed(4))
    assert torch.isfinite(lp).all()
    assert_moves_legal(positions, range(len(positions)), t, a)


# ---- reproducibility, purity, edge cases --------------------------------------------------------

def test_generator_makes_sampling_reproducible(positions):
    _, mask, phases = tensors(positions)
    logits = make_logits("normal", mask, seed=51)
    first = actions.sample(logits, mask, phases, generator=torch.Generator().manual_seed(77))
    second = actions.sample(logits, mask, phases, generator=torch.Generator().manual_seed(77))
    for x, y in zip(first, second):
        assert torch.equal(x, y)
    other = actions.sample(logits, mask, phases, generator=torch.Generator().manual_seed(78))
    assert not (torch.equal(first[0], other[0]) and torch.equal(first[1], other[1]))


def test_no_generator_uses_global_rng(positions):
    _, mask, phases = tensors(positions)
    logits = make_logits("normal", mask, seed=52)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        first = actions.sample(logits, mask, phases)
        torch.manual_seed(5)
        second = actions.sample(logits, mask, phases)
    for x, y in zip(first, second):
        assert torch.equal(x, y)


def test_inputs_are_not_modified(positions):
    obs, mask, phases = tensors(positions)
    logits = make_logits("huge_illegal", mask, seed=53)
    types = torch.tensor([moves[-1][0] for moves in positions.legal])
    args = torch.tensor([moves[-1][1] for moves in positions.legal])
    saved = [x.clone() for x in (obs, mask, phases, logits, types, args)]
    bool_mask = mask != 0
    saved_bool = bool_mask.clone()
    actions.phase_of(obs)
    actions.arg_sections(types, phases)
    actions.masked_logits(logits, mask)
    actions.masked_logits(logits, bool_mask)
    actions.sample(logits, mask, phases, generator=torch.Generator().manual_seed(0))
    actions.mode(logits, bool_mask, phases)
    actions.log_prob_entropy(logits, mask, phases, types, args)
    for before, after in zip(saved, (obs, mask, phases, logits, types, args)):
        assert torch.equal(before, after)
    assert torch.equal(saved_bool, bool_mask)


def test_empty_batch():
    logits = torch.zeros(0, MASK_SIZE)
    mask = torch.zeros(0, MASK_SIZE, dtype=torch.uint8)
    phases = torch.zeros(0, dtype=torch.int64)
    types, args, log_prob = actions.sample(logits, mask, phases,
                                           generator=torch.Generator().manual_seed(0))
    assert types.shape == args.shape == log_prob.shape == (0,)
    assert types.dtype == args.dtype == torch.int64
    t, a = actions.mode(logits, mask, phases)
    assert t.shape == a.shape == (0,)
    lp, h = actions.log_prob_entropy(logits, mask, phases, types, args)
    assert lp.shape == h.shape == (0,)
    assert actions.masked_logits(logits, mask).shape == (0, MASK_SIZE)
    assert actions.arg_sections(types, phases).shape == (0,)


def test_bool_and_uint8_masks_agree(positions):
    _, mask, phases = tensors(positions)
    logits = make_logits("normal", mask, seed=61)
    results = []
    for m in (mask, mask != 0, mask * 255):
        s = actions.sample(logits, m, phases, generator=torch.Generator().manual_seed(8))
        md = actions.mode(logits, m, phases)
        lpe = actions.log_prob_entropy(logits, m, phases, s[0], s[1])
        results.append((*s, *md, *lpe))
    for other in results[1:]:
        for x, y in zip(results[0], other):
            assert torch.equal(x, y)
