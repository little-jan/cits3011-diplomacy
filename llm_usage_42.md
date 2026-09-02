# LLM prompts used in this project

1.

My task is to design an agent to play diplomacy against other pre-written agents
The strategy for my agent is to:
- Score the easiest-to-obtain supply centre (based on distance and occupied/vacancy)
- Score the rate of support (for another army/fleet)
- Score the rate of move
Combine these three scores above into a value rating function to dictate the next order for the agent to execute
These scores are dependent on the season, as well as the gauge of incoming opponents

Can you structure an agent to play diplomacy based on these strategies? Let me know if you need more information about constraints or how something should be implemented