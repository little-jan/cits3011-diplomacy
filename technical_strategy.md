# From Gemini (prompt 1)

Here is a structural architecture for your Diplomacy agent. Your strategy maps perfectly to a **heuristic-based utility agent**, where every possible valid order for a unit is generated, scored against your three criteria, and the set of orders with the highest combined utility is executed.

Here is how to structure the agent's logic, broken down into its evaluation pipeline and state modifiers.

## 1. The Value Rating Function (VRF) Architecture

For every unit you control, the agent will generate a list of legal actions ($A$) (Hold, Move, Support Hold, Support Move, Convoy). Each action receives a score based on your three criteria, modified by dynamic weights.

The core equation for your agent is:


$$VRF(A) = W_{SC}(\text{Season}, \text{Threat}) \cdot S_{SC}(A) + W_{Sup}(\text{Season}, \text{Threat}) \cdot S_{Sup}(A) + W_{Mov}(\text{Season}, \text{Threat}) \cdot S_{Mov}(A)$$

The action with the highest $VRF$ is selected. (Note: You must also resolve conflicts between your own units, ensuring two units don't try to move to the same highly-scored province without supporting each other).

## 2. Defining the Sub-Scores

### A. Easiest-to-Obtain Supply Center ($S_{SC}$)

This evaluates how effectively an action moves a unit toward a conquerable SC.

* **Distance Metric:** Use a Breadth-First Search (BFS) or Dijkstra's algorithm from the unit's current province to all unowned/enemy SCs.
* **Vacancy/Occupation Modifier:**
* Empty neutral SC: High multiplier.
* Empty enemy SC: High multiplier.
* Occupied SC: Lower multiplier, inversely proportional to the defensive support that enemy unit has.


* **Action Mapping:** If the action is a `Move` that decreases the path distance to the highest-value SC, it receives a high $S_{SC}$ score.

### B. Rate of Support ($S_{Sup}$)

This evaluates the utility of helping another unit (either for attack or defense).

* **Offensive Support:** If an allied unit is moving into an occupied SC, issuing a `Support Move` order gets a score equal to the $S_{SC}$ value of the province being attacked.
* **Defensive Support:** If an allied unit is holding a crucial SC and the "gauge of incoming opponents" is high (e.g., 2+ enemy units adjacent to it), issuing a `Support Hold` order gets a massive score spike to prevent dislodgement.
* **Action Mapping:** `Support` orders only receive a score $> 0$ if the supported unit is actually executing the corresponding move/hold.

### C. Rate of Move ($S_{Mov}$)

This acts as your tactical positioning and threat response score, separate from direct SC capture.

* **Chokepoint Control:** Moving into highly connected provinces (like the Mid-Atlantic Ocean, Black Sea, or Burgundy) scores high for $S_{Mov}$ because it increases future mobility.
* **Threat Response:** If the "gauge of incoming opponents" shows enemies massing near your borders, moving units toward those defensive lines increases $S_{Mov}$.
* **Action Mapping:** Standard `Move` orders to non-SC provinces.

## 3. Dynamic Modifiers (Season & Threat)

Your weights ($W_{SC}, W_{Sup}, W_{Mov}$) must shift dynamically based on the game state.

**The Season Modifier:**

* **Spring Phases:** $W_{Mov}$ and $W_{Sup}$ are prioritized. Spring is about positioning and bouncing enemies. You cannot capture an SC in the Spring, so $W_{SC}$ is strictly about moving *adjacent* to an SC.
* **Fall Phases:** $W_{SC}$ becomes the dominant weight. The agent must prioritize actions that actually occupy the SC province this turn to claim it during the Winter phase.
* **Winter (Build/Disband):** The VRF is bypassed. The agent simply builds units in home SCs closest to the highest concentration of enemy threats, or disbands units furthest from the front lines.

**Gauge of Incoming Opponents (Threat Level):**
You can calculate a "Threat Score" for every province you own by counting the number of adjacent enemy units vs. adjacent friendly units.

* If **Threat is High** (Enemy outnumbers you locally): $W_{Sup}$ (Defensive) and $W_{Mov}$ (Retreating/Regrouping) increase.
* If **Threat is Low** (You outnumber the enemy): $W_{SC}$ and $W_{Sup}$ (Offensive) increase to exploit the weakness.

## Implementation Constraints to Consider

To build this accurately, I need a bit more information on your specific technical setup.