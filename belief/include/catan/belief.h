#pragma once
#include <array>
#include <cstdint>
#include <type_traits>

#include "catan/board.h"
#include "catan/event.h"
#include "catan/state.h"

// What each player can know about the cards they cannot see. Lives outside `core` because it
// models knowledge, not rules: it is fed only the public state and the event stream, applying
// each event as every observer would see it (a STEAL's resource only to thief and victim, a
// DEV_BOUGHT's card only to the buyer).
//
// Resource hands are tracked per (observer, target) as cards of known type plus a count of
// unknown ones. Updates never exclude the truth (soundness) but may forget information: when a
// card of unseen type leaves a hand (a steal seen by a non-participant), every known count of
// that hand drops by one, the dropped cards becoming unknown, since any of them may have been
// the stolen one. If the hand held a single type and nothing unknown, the type is deduced.
//
// Monopoly removes the victim's cards of that type known-first; the fact that the victim now
// holds none of it is not representable and is forgotten.
//
// Development cards in an opponent's hand are always unknown (dev_count is public). What is
// public is how many of each type have been played, how many the current player bought this
// turn, and each player's public VP when they last ended a turn, which bounds their hidden VP
// cards: a player holding 10 points on their own turn would already have won.

namespace catan {

struct HandBelief {
    std::array<uint8_t, NUM_RES> known;  // cards whose type the observer knows
    uint8_t unknown;                     // known + unknown == hand_size
};

struct BeliefState {
    // hand[o][t]: observer o's view of target t's resources. hand[o][o] is exact.
    std::array<std::array<HandBelief, NUM_PLAYERS>, NUM_PLAYERS> hand;
    std::array<uint8_t, NUM_DEV> dev_played;  // public, all players
    uint8_t bought_this_turn;                 // dev cards bought by the current player this turn
    // Public VP of each player right after their last END_TURN (0 before their first turn).
    // While t is not the current player, t's hidden VP cards are < 10 - vp_at_turn_end[t].
    std::array<uint8_t, NUM_PLAYERS> vp_at_turn_end;
    uint8_t reserved_[2]{};
};

static_assert(std::is_trivially_copyable_v<BeliefState>);
static_assert(std::has_unique_object_representations_v<BeliefState>);

// Everyone knows everything: all hands empty at the start of setup.
BeliefState initial_belief();

// Updates every observer's belief with the events of one apply/apply_chance. `after` is the
// public state those events produced (for hand sizes and VP at TURN_ENDED).
void observe(BeliefState& belief, const PublicState& after, const EventList& events);

// Soundness check against the true state: for every observer o and target t, known + unknown
// == hand_size[t], the truth holds at least `known` of each resource, hand[o][o] is exact, the
// dev counters match (played + held + deck == deck composition, knights against knights_played,
// bought_this_turn outside TERMINAL), and every player's VP cards respect the VP bounds that
// resample_hidden enforces. With `observer` set, only that observer's row of `hand` is checked
// (the public parts still are): the guarantee resample_hidden gives. Returns nullptr if sound,
// else a static description.
const char* belief_violation(const BeliefState& belief, const GameState& truth,
                             uint8_t observer = NONE);

// Unseen resource cards from o's point of view, per type: 19 minus the bank minus every card o
// knows the type of (its own hand included).
std::array<uint8_t, NUM_RES> unseen_resources(const BeliefState& belief, const PublicState& pub,
                                              uint8_t observer);

// Encoder bounds on how many of resource r target t holds, as observer o sees it.
// min = known, max = known + min(unknown, unseen).
uint8_t hand_min(const BeliefState& belief, uint8_t observer, uint8_t target, int r);
uint8_t hand_max(const BeliefState& belief, const PublicState& pub, uint8_t observer,
                 uint8_t target, int r);

// A full state the observer cannot tell apart from `truth`: same public state, same observer
// hidden state, and opponents' unknown resource cards, opponents' dev cards and the deck dealt
// at random from what the observer has not seen. Deterministic in `seed` (counter-based RNG,
// Stream::RESAMPLE). Guarantees: invariant_violation(board, result) == nullptr,
// belief_violation(belief, result, observer) == nullptr, play can continue from the result, and no
// opponent holds hidden VP cards that would already have won (for the current player:
// public_vp + VP cards < 10 now and vp_at_turn_end + VP cards held before this turn < 10; for
// others: vp_at_turn_end + VP cards < 10; a TERMINAL state's winner is unconstrained).
// The current player's dev_bought_this_turn (when not the observer) is assigned from their
// dealt cards so its total equals bought_this_turn.
//
// Publicly implied hand constraints on the opponent to move are enforced too: a discarder past
// their first card holds at least pending_discard cards at or above discard_floor, and a player
// in EXCHANGE_RECV holds at least their harbor rate of exchange_give (hence `board`).
//
// Deals are uniform conditioned on these constraints (rejection, up to 64 tries); if every try
// fails, the last deal is repaired by deterministic swaps that move constrained cards between
// holders, which always succeeds because the truth is a feasible deal.
GameState resample_hidden(const Board& board, const GameState& truth, const BeliefState& belief,
                          uint8_t observer, uint64_t seed);

// Other observers' rows of `belief` describe the real hands, not the resampled ones, and would
// go unsound (and underflow) if observe() kept running on them inside a determinization. This
// returns a belief sound for `resampled`: every row is the observer's row, with each player's
// own hand made exact. Other players thereby "know" what the observer knows, which is an
// approximation of their knowledge, never a contradiction of the dealt state.
BeliefState resampled_belief(const BeliefState& belief, const GameState& resampled,
                             uint8_t observer);

}  // namespace catan
