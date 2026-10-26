#include "catan/rules.h"

#include "catan/mask.h"
#include "rules_internal.h"

using namespace std;

namespace catan {

using namespace detail;

namespace {

[[noreturn]] void unimplemented(Phase phase) {
    detail::unimplemented("phase", static_cast<int>(phase));
}
[[noreturn]] void unimplemented(ChanceKind kind) {
    detail::unimplemented("chance kind", static_cast<int>(kind));
}

Decision chance_decision(const GameState& s) {
    Decision d{};
    d.kind = DecisionKind::CHANCE;
    d.phase = s.pub.phase;
    const auto kind = static_cast<ChanceKind>(s.pub.pending_chance);
    switch (kind) {
        case ChanceKind::DICE: d.chance = dice_dist(); return d;
        case ChanceKind::STEAL: d.chance = steal_dist(s); return d;
        case ChanceKind::DEV_DRAW: d.chance = dev_draw_dist(s); return d;
        case ChanceKind::NONE: unimplemented(kind);
    }
    unimplemented(kind);
}

}  // namespace

Decision next_decision(const Board& board, const GameState& state) {
    const PublicState& pub = state.pub;
    if (pub.pending_chance != 0) return chance_decision(state);

    switch (pub.phase) {
        case Phase::SETUP: return setup_decision(pub);
        case Phase::ROLL: return roll_decision(state);
        case Phase::MAIN: return main_decision(board, state);
        case Phase::EXCHANGE_RECV: return exchange_recv_decision(board, state);
        case Phase::DISCARD: return discard_decision(state);
        case Phase::MOVE_ROBBER: return move_robber_decision(state);
        case Phase::STEAL: return steal_decision(board, state);
        case Phase::ROAD_BUILDING: return road_building_decision(state);
        case Phase::YOP_PICK: return yop_decision(state);
        case Phase::MONOPOLY_PICK: return monopoly_decision(state);
        case Phase::TERMINAL: {
            Decision d{};
            d.kind = DecisionKind::TERMINAL;
            d.phase = Phase::TERMINAL;
            return d;
        }
    }
    unimplemented(pub.phase);
}

EventList apply(const Board& board, GameState& state, Action action) {
    CATAN_CHECK(state.pub.pending_chance == 0);
    CATAN_CHECK(is_legal(next_decision(board, state).mask, state.pub.phase, action));

    EventList events;
    switch (state.pub.phase) {
        case Phase::SETUP: apply_setup(board, state, action, events); break;
        case Phase::ROLL: apply_roll(state, action, events); break;
        case Phase::MAIN: apply_main(board, state, action, events); break;
        case Phase::EXCHANGE_RECV: apply_exchange_recv(board, state, action, events); break;
        case Phase::DISCARD: apply_discard(state, action, events); break;
        case Phase::MOVE_ROBBER: apply_move_robber(board, state, action, events); break;
        case Phase::STEAL: apply_steal(state, action, events); break;
        case Phase::ROAD_BUILDING: apply_road_building(state, action, events); break;
        case Phase::YOP_PICK: apply_yop(state, action, events); break;
        case Phase::MONOPOLY_PICK: apply_monopoly(state, action, events); break;
        case Phase::TERMINAL: unimplemented(state.pub.phase);  // TERMINAL has no actions
    }
    if (state.pub.phase != Phase::SETUP) check_victory(state, events);
    state.pub.ply = static_cast<uint16_t>(state.pub.ply + 1);
    return events;
}

EventList apply_chance(const Board& board, GameState& state, uint8_t outcome) {
    CATAN_CHECK(state.pub.pending_chance != 0);

    // Cleared before dispatch, so a handler that owes another chance can set it again.
    const auto kind = static_cast<ChanceKind>(state.pub.pending_chance);
    state.pub.pending_chance = 0;

    EventList events;
    switch (kind) {
        case ChanceKind::DICE: apply_dice(board, state, outcome, events); break;
        case ChanceKind::STEAL: apply_steal_outcome(state, outcome, events); break;
        case ChanceKind::DEV_DRAW: apply_dev_draw(state, outcome, events); break;
        case ChanceKind::NONE: unimplemented(kind);
    }
    check_victory(state, events);
    state.pub.ply = static_cast<uint16_t>(state.pub.ply + 1);
    return events;
}

}  // namespace catan
