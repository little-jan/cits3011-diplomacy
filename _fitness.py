PARAM_NAMES = [
    'GAMMA', 'MAX_D', 'COMBAT_K', 'COMBAT_EDGE', 'GRAD_W', 
    'CAPTURE_F', 'CAPTURE_S', 'DEFEND_F', 'DEFEND_S', 
    'REVERSE_PEN', 'BOUNCE_PEN', 'SELF_BOUNCE_PEN', 
    'BLOCK_PEN', 'IDLE_SUP_PEN', 'QUIET_SUP_PEN'
]


BOUNDS = [
    (0.50, 0.99),  # GAMMA
    (5.00, 20.0),  # MAX_D (will be cast to int)
    (1.00, 5.00),  # COMBAT_K
    (0.10, 1.00),  # COMBAT_EDGE
    (0.50, 3.00),  # GRAD_W
    (1.00, 5.00),  # CAPTURE_F
    (0.10, 3.00),  # CAPTURE_S
    (0.50, 3.00),  # DEFEND_F
    (0.10, 2.00),  # DEFEND_S
    (0.00, 1.00),  # REVERSE_PEN
    (0.00, 1.00),  # BOUNCE_PEN
    (0.00, 2.00),  # SELF_BOUNCE_PEN
    (0.00, 2.00),  # BLOCK_PEN
    (0.00, 1.00),  # IDLE_SUP_PEN
    (0.00, 1.00)   # QUIET_SUP_PEN
]


from test_42 import experiment, POOL_2
from agent_42 import StudentAgent

def fitness_fn(theta):
    params = list(theta)
    params[1] = int(round(params[1]))
    
    def agent_factory():
        agent = StudentAgent()
        
        agent.GAMMA = params[0]
        agent.MAX_D = params[1]  
        agent.COMBAT_K = params[2]
        agent.COMBAT_EDGE = params[3]
        agent.GRAD_W = params[4]
        agent.CAPTURE_F = params[5]
        agent.CAPTURE_S = params[6]
        agent.DEFEND_F = params[7]
        agent.DEFEND_S = params[8]
        agent.REVERSE_PEN = params[9]
        agent.BOUNCE_PEN = params[10]
        agent.SELF_BOUNCE_PEN = params[11]
        agent.BLOCK_PEN = params[12]
        agent.IDLE_SUP_PEN = params[13]
        agent.QUIET_SUP_PEN = params[14]
        
        return agent

    avg_sc, std_sc, win_rate = experiment(
        player_agent=agent_factory, 
        opponent_agent_pool=POOL_2, 
        repeat_nums=3, 
        verbose=False
    )
    
    return win_rate