import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from game import run_one_game
from agent_baselines import StaticAgent, RandomAgent, GreedyAgent, AttitudeAgent
from agent_42 import StudentAgent

'''
Group 42 experiments.

`experiment` reproduces the marking protocol (the player agent takes each of
the seven powers in turn).  `ablation` re-runs the same protocol with one
component of the agent disabled, which is how the quantitative results in the
report were produced.
'''

ALL_POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']

POOL_1 = [StaticAgent]
POOL_2 = [RandomAgent, AttitudeAgent, AttitudeAgent, GreedyAgent, GreedyAgent]


def scoring(centres):
    scores = {k: min(v, 18) for k, v in centres.items()}
    wins = {}
    for k, v in scores.items():
        if v == 18:
            wins[k] = 'WIN'
        elif v == 0:
            wins[k] = 'DEFEAT'
        else:
            wins[k] = 'SURVIVE'
    return scores, wins


def experiment(player_agent, opponent_agent_pool, repeat_nums=10, verbose=True):
    all_scores = defaultdict(list)
    all_wins = defaultdict(list)
    total_games = repeat_nums * len(ALL_POWERS)

    with tqdm(total=total_games, leave=False) as pbar:
        for _ in range(repeat_nums):
            for i in ALL_POWERS:
                agents_dict = {}
                for p in ALL_POWERS:
                    if p == i:
                        agents_dict[p] = player_agent()
                    else:
                        agents_dict[p] = random.choice(opponent_agent_pool)()
                results, _ = run_one_game(agents_dict)
                scores, wins = scoring(results)
                all_scores[i].append(scores[i])
                all_scores['ALL'].append(scores[i])
                all_wins[i].append(wins[i])
                all_wins['ALL'].append(wins[i])
                pbar.update(1)

    scores_avg = {k: np.mean(v) for k, v in all_scores.items()}
    scores_std = {k: np.std(v) for k, v in all_scores.items()}
    win_rates, survive_rates, defeat_rates = {}, {}, {}
    for k, v in all_wins.items():
        win_rates[k] = round(100 * sum(i == 'WIN' for i in v) / len(v), 2)
        survive_rates[k] = round(100 * sum(i == 'SURVIVE' for i in v) / len(v), 2)
        defeat_rates[k] = round(100 * sum(i == 'DEFEAT' for i in v) / len(v), 2)

    if verbose:
        print('----- Per-Power and Overall Performance of the Player Agent -----')
        for p in ALL_POWERS + ['ALL']:
            print(f'{p}: SCs - {round(scores_avg[p], 2)}+-{round(scores_std[p], 2)}, '
                  f'Wins - {win_rates[p]}%, Survives - {survive_rates[p]}%, '
                  f'Defeats - {defeat_rates[p]}%')
        print('-----------------------------------------------------------------')

    return scores_avg['ALL'], scores_std['ALL'], win_rates['ALL']


# --------------------------------------------------------------------------- #
#  Ablation study
# --------------------------------------------------------------------------- #

# Each entry disables exactly one component of the final agent.
ABLATIONS = [
    ('Full agent',                     {}),
    ('- local search (greedy init)',   {'use_local_search': False}),
    ('- potential field (T1)',         {'use_diffusion': False}),
    ('- pairwise neighbourhood (T2)',  {'use_pair_moves': False}),
    ('- opponent model (T3)',          {'use_opponent_model': False}),
    ('- season/adjustment (T4)',       {'use_season': False, 'use_adjustment': False}),
]


def ablation(pool, repeat_nums=3, label='', seed=0, subset=None):
    print(f'===== Ablation study {label} (n={repeat_nums * 7} games each) =====')
    rows = []
    for name, kwargs in ABLATIONS:
        if subset is not None and name not in subset:
            continue
        random.seed(seed)          # identical opponent draws for every variant
        factory = (lambda kw=kwargs: StudentAgent(**kw))
        avg, std, win = experiment(factory, pool, repeat_nums, verbose=False)
        rows.append((name, avg, std, win))
        print(f'{name:<32} SC {avg:5.2f} +- {std:4.2f}   win {win:5.1f}%')
    print()
    return rows


if __name__ == '__main__':
    print('Evaluating Scenario 1 ...')
    experiment(StudentAgent, POOL_1, repeat_nums=10)

    print('Evaluating Scenario 2 ...')
    experiment(StudentAgent, POOL_2, repeat_nums=10)

    ablation(POOL_1, repeat_nums=3, label='Scenario 1')
    ablation(POOL_2, repeat_nums=3, label='Scenario 2')
