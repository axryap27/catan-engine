// Plays one game with uniformly random legal moves and prints the event log.
//   catan_sim [seed] [--quiet] [--board] [--ids] [--watch]
//     --quiet   print only the final summary line
//     --board   after the game, print the final position (god view: every hand and dev card)
//     --ids     label empty corners and edges with their vertex and edge ids in every board
//               printed by --board and --watch
//     --watch   print the position after every turn
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "catan/board.h"
#include "catan/chance.h"
#include "catan/rng.h"
#include "catan/rules.h"
#include "catan/text.h"

using namespace std;
using namespace catan;

namespace {

// Dice rolls, turn ends and the end of the game stand out; everything else is indented
// under the roll that caused it.
void print(const Event& e) {
    const bool top = e.type == EventType::DICE_ROLLED || e.type == EventType::TURN_ENDED ||
                     e.type == EventType::GAME_OVER;
    printf("%s%s\n", top ? "" : "  ", describe_event(e).c_str());
}

void usage() {
    fprintf(stderr, "usage: catan_sim [seed] [--quiet] [--board] [--ids] [--watch]\n");
}

}  // namespace

int main(int argc, char** argv) {
    uint64_t seed = 1;
    bool quiet = false, show_board = false, watch = false;
    RenderOptions render;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--quiet") == 0) quiet = true;
        else if (strcmp(argv[i], "--board") == 0) show_board = true;
        else if (strcmp(argv[i], "--watch") == 0) watch = true;
        else if (strcmp(argv[i], "--ids") == 0) render.vertex_ids = render.edge_ids = true;
        else if (strncmp(argv[i], "--", 2) == 0) {
            usage();
            return 2;
        } else seed = strtoull(argv[i], nullptr, 10);
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
        for (const Event& e : events) {
            if (!quiet) print(e);
            if (watch && e.type == EventType::TURN_ENDED) {
                printf("%s\n\n", render_state(board, s, NONE, render).c_str());
            }
        }
    }

    printf("seed %llu: %d turns, %d decisions, VP", static_cast<unsigned long long>(seed),
           s.pub.turn, s.pub.ply);
    for (int p = 0; p < NUM_PLAYERS; ++p) printf(" P%d=%d", p, s.pub.public_vp[p]);
    printf("\n");
    if (show_board) printf("\n%s\n", render_state(board, s, NONE, render).c_str());
    return 0;
}
