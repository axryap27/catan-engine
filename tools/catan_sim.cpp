// Plays one game with uniformly random legal moves and prints the event log.
//   catan_sim [seed] [--quiet]
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "catan/board.h"
#include "catan/chance.h"
#include "catan/rng.h"
#include "catan/rules.h"

using namespace std;
using namespace catan;

namespace {

const char* const RES[] = {"brick", "lumber", "wool", "grain", "ore"};
const char* const DEV[] = {"knight", "victory point", "road building", "year of plenty",
                           "monopoly"};

void print(const Event& e) {
    const int p = e.player;
    switch (e.type) {
        case EventType::SETTLEMENT_BUILT: printf("  P%d builds a settlement at v%d\n", p, e.a); break;
        case EventType::CITY_BUILT: printf("  P%d builds a city at v%d\n", p, e.a); break;
        case EventType::ROAD_BUILT: printf("  P%d builds a road at e%d\n", p, e.a); break;
        case EventType::GAIN: printf("  P%d +%d %s\n", p, e.b, RES[e.a]); break;
        case EventType::SPEND: printf("  P%d -%d %s\n", p, e.b, RES[e.a]); break;
        case EventType::DISCARD: printf("  P%d discards %s\n", p, RES[e.a]); break;
        case EventType::DICE_ROLLED: printf("P%d rolls %d\n", p, e.a); break;
        case EventType::ROBBER_MOVED: printf("  P%d moves the robber to hex %d\n", p, e.a); break;
        case EventType::STEAL: printf("  P%d steals %s from P%d\n", p, RES[e.a], e.target); break;
        case EventType::DEV_BOUGHT: printf("  P%d buys a %s card\n", p, DEV[e.a]); break;
        case EventType::DEV_PLAYED: printf("  P%d plays %s\n", p, DEV[e.a]); break;
        case EventType::MONOPOLY:
            if (e.b > 0) printf("  P%d takes %d %s from P%d\n", p, e.b, RES[e.a], e.target);
            break;
        case EventType::LONGEST_ROAD:
            if (p == NONE) printf("  Longest Road is now unclaimed\n");
            else printf("  P%d takes Longest Road\n", p);
            break;
        case EventType::LARGEST_ARMY: printf("  P%d takes Largest Army\n", p); break;
        case EventType::GAME_OVER:
            if (e.player == NONE) printf("Game over: turn cap reached\n");
            else printf("Game over: P%d wins\n", p);
            break;
        default: break;
    }
}

}  // namespace

int main(int argc, char** argv) {
    uint64_t seed = 1;
    bool quiet = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--quiet") == 0) quiet = true;
        else seed = strtoull(argv[i], nullptr, 10);
    }

    const Board board = random_board(seed);
    GameState s = initial_state(board);
    uint64_t moves = 0;

    for (Decision d = next_decision(board, s); d.kind != DecisionKind::TERMINAL;
         d = next_decision(board, s)) {
        EventList events;
        if (d.kind == DecisionKind::CHANCE) {
            events = apply_chance(board, s, sample_chance(d.chance, s, seed));
        } else {
            const auto n = random_below(seed, 1000, moves++, d.n_legal);
            events = apply(board, s, nth_legal(d.mask, d.phase, static_cast<int>(n)));
        }
        if (!quiet) {
            for (const Event& e : events) print(e);
        }
    }

    printf("seed %llu: %d turns, %d decisions, VP", static_cast<unsigned long long>(seed),
           s.pub.turn, s.pub.ply);
    for (int p = 0; p < NUM_PLAYERS; ++p) printf(" P%d=%d", p, s.pub.public_vp[p]);
    printf("\n");
    return 0;
}
