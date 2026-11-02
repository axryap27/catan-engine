"""PPO trainer for the Catan engine.

    engine    imports the compiled catan_engine module (from build/python/bind if needed)
    actions   the move distribution: a type, then that type's argument, both masked
    policy    PolicyNet (observation -> logits) and Agent (plays games: numpy in, numpy out)
    evaluate  the random bot and play_match: win rate against random, greedy or itself
    rollout   play(): batches of games against bots, optionally recording the agent's moves
    reinforce stage 2: REINFORCE against bots (Config, Trainer)
"""
