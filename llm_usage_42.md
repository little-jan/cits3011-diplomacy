# LLM prompts used in this project

## 1.

My task is to design an agent to play diplomacy against other pre-written agents.
The strategy for my agent is to:

- Score the easiest-to-obtain supply centre (based on distance and occupied/vacancy)
- Score the rate of support (for another army/fleet)
- Score the rate of move

Combine these three scores above into a value rating function to dictate the next order for the agent to execute. These scores are dependent on the season, as well as the gauge of incoming opponents.

Can you structure an agent to play diplomacy based on these strategies? Let me know if you need more information about constraints or how something should be implemented.

**Zeel:**
Based on the D. Norman DumbBot given in the research paper source of the neighbour-averaging province-valuation idea, we will use a local search and a hill-climbing idea, create a strategy, with following constraints and rules:

Your agent will play the game against baseline agents developed by the teaching staff. There are five different baseline agents.

- **Static Agent**: This is an agent that always takes the default actions, i.e., hold.
- **Random Agent**: This is an agent that always takes random actions.
- **Attitude Agent**: This is an agent that takes random actions, but has attitudes towards other powers, including being friendly, neutral, or hostile. The attitude depends on other players' actions and can change during the game. A friendly agent will never attack you, a hostile one will never support you, and a neutral one can do anything.
- **Greedy Agent**: This is an agent that always takes greedy actions, without long-term planning. Each unit controlled by the agent will move towards and attack the closest supply centre, or support other units if having the same target.
- **Hidden Agent**: This is an unknown agent.

### Scenarios

- **Scenario 1**: Your agent will control a random power. Other powers are all controlled by copies of Static Agent.
- **Scenario 2**: Your agent will control a random power. Other powers are controlled by copies of agents randomly chosen from Random Agent, Attitude Agent, and Greedy Agent. The Random Agent is less likely to appear than the other two.
- **Scenario 3**: Your agent will control a random power. Other powers are controlled by copies of agents randomly chosen from Random Agent, Attitude Agent, Greedy Agent, and Hidden Agent. There will be exactly one Hidden Agent in each game. The Hidden Agent is a reasonably strong agent with a ~50% win rate in Scenario 2.
- **Scenario 4**: All the group agents will be put together to play a multi-round tournament.

### Marking Rubrics

The marking of the agent and the report will be independent of each other. The marking of the agent will focus on the performance. The marking of the report will focus on the knowledge, thinking, reasoning, and presentation.

**Agent Rubrics (15 pts)[1]:**

- **Scenario 1 (5 pts)**
  - The agent achieves >2% win rate, or captures >7 supply centres on average. (1 pt)
  - The agent achieves >20% win rate, or captures >12 supply centres on average. (3 pts)
  - The agent achieves >90% win rate, or captures >16 supply centres on average. (5 pts)
- **Scenario 2 (5 pts)**
  - The agent achieves >2% win rate, or captures >7 supply centres on average. (1 pt)
  - The agent achieves >25% win rate, or captures >10 supply centres on average. (3 pts)
  - The agent achieves >50% win rate, or captures >13 supply centres on average. (5 pts)
- **Scenario 3 (5 pts)**
  - The agent achieves >2% win rate, or captures >7 supply centres on average. (1 pt)
  - The agent achieves >20% win rate, or captures >9 supply centres on average. (3 pts)
  - The agent achieves >40% win rate, or captures >12 supply centres on average. (5 pts)
- **Scenario 4 (Bonus)[2]**
  - The agent ranks top 3 among all group agents in Scenario 4. (3 bonus pts)

Ignore the hidden agent scenario for now.

---

Create a detailed explanation of each probabilistic function and order set calculation; dry run the whole strategy.

---

Implement the lookup strategy and incoming calculation function, `def _potential`. Don't modify the joint order logic or graph-building functions.

---

For each tunable constant except for time budget, run a few iterations of the game and give the best possible combination for testing (threshold: 65% win rate in Scenario 1 and Scenario 2).
