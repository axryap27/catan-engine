#!/usr/bin/env python3
"""Play Catan in the terminal: you against three greedy bots.

    python bind/python/play.py [--seed N] [--seat S] [--auto] [--standard]

Needs only the catan_engine module on PYTHONPATH (e.g. build/python/bind). On your turn the board
is printed as your seat sees it, then the numbered legal actions; type a number, `b` to reprint
the board or `q` to quit. With --auto every seat is a greedy bot and the game plays out with no
input (events and the final board are printed from the god view).
"""
import argparse

import catan_engine as ce

SETTLEMENT, CITY, ROAD = (int(ce.ActionType.BUILD_SETTLEMENT), int(ce.ActionType.BUILD_CITY),
                          int(ce.ActionType.BUILD_ROAD))


def show_board(game, seat, actions):
    """The board with vertex / edge ids when one of the legal actions needs them."""
    kinds = {t for t, _ in actions}
    print(game.render(seat, vertex_ids=bool(kinds & {SETTLEMENT, CITY}), edge_ids=ROAD in kinds))


def ask(game, seat):
    """Prints the position and returns the chosen (type, arg), or None to quit."""
    actions = game.legal_actions()
    show_board(game, seat, actions)
    print(f"\nP{seat} to act ({game.phase.name.lower()}), turn {game.turn}:")
    for i, (t, a) in enumerate(actions):
        print(f"  {i:3d}  {game.describe(t, a)}")
    while True:
        try:
            text = input("choose [number, b = board, q = quit]> ").strip().lower()
        except EOFError:
            return None
        if text == "q":
            return None
        if text == "b":
            show_board(game, seat, actions)
        elif text.isdigit() and int(text) < len(actions):
            return actions[int(text)]
        elif text == "" and len(actions) == 1:  # forced move: just press enter
            return actions[0]
        else:
            print("not a legal choice")


def main():
    ap = argparse.ArgumentParser(description="Play Catan against three greedy bots.")
    ap.add_argument("--seed", type=int, default=1, help="board and dice seed (default 1)")
    ap.add_argument("--seat", type=int, default=0, choices=range(ce.NUM_PLAYERS),
                    help="your seat (default 0)")
    ap.add_argument("--auto", action="store_true", help="all seats greedy, no input")
    ap.add_argument("--standard", action="store_true", help="the standard beginner board")
    args = ap.parse_args()

    game = ce.Game(args.seed, standard_board=args.standard)
    viewer = None if args.auto else args.seat  # None: god view
    print(f"seed {args.seed}; you are P{args.seat}" if not args.auto else f"seed {args.seed}")

    while not game.done:
        seat = game.decider
        if args.auto or seat != args.seat:
            action = game.greedy_action()
        else:
            action = ask(game, seat)
            if action is None:
                print("bye")
                return
        for event in game.step(*action):
            print("  " + event.describe(viewer))

    print()
    print(game.render(viewer))
    vp = " ".join(f"P{p}={v}" for p, v in enumerate(game.vp))
    if game.winner is None:
        print(f"\nNo winner: the turn cap ended the game ({vp}).")
    else:
        you = " You win!" if game.winner == args.seat and not args.auto else ""
        print(f"\nP{game.winner} wins after {game.turn} turns ({vp}).{you}")


if __name__ == "__main__":
    main()
