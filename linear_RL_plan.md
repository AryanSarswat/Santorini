### **Overview/Goals** 
---
1. Create an AI to play Santorini, making use of:
    * Linear Value Function Approximator
    * Minimax Search (optionally include AB pruning)
    * Various RL Methods
        * Start simple with basic TD learning (might not work well, but can try implementing first)(learn value fn by TD learning then use value function in minimax search)
        * followed by TD lambda, TD root, TD leaf, Treestrap
        * then try methods that incorporate RL with minimax search (strikes me as being somewhat related to Q-learning, in that they _could_ behave one way but update the value function approximator with the **best** value)
2. Evaluate relative performance of AI (efficacy of using linear fn approximator + minimax as a whole?)
    * First comparing against an AI that acts randomly
    * Then, against AIs created by other group members (e.g. using MCTS)
3. I would also like to investigate the relative desirability of diff RL training methods (e.g. how quickly they train, their performance after a fixed number of rounds).
    * A promising one would be combining RL training with minimax search; i.e. the function approximator will attempt to learn the minimax values, which will hopefully be more efficient.
4. **Stretch Goal**: One of the main drawbacks of using a linear function approximator would be the need to come up with 'good' features of the value of a certain game state. These require _heuristics and prior knowledge_ about the game itself, and doesn't really generalize to new situations (e.g. a different game). **How can we resolve this problem?**
    * I think one way would be to find a way to 'randomly' invent patterns from the game board (with a limited number of squares) and 'try' them out as features.
    * Using a method of quantifying relative feature importance (perhaps the size of their weights), we can then trim out the 'useless' features.
    * However, I am not sure how feasible this idea is computationally, and implementation-wise.
    * Needs more thought and research into whether there has been research into such approaches - look into Logistello
    * Additionally, at this point you might as well consider going with a **convolutional neural network** or something....although that comes with drawbacks regarding probable convergence issues and lower explainability.

---
### **Tasks and Timeline**
---
* **Week 7 (midterms  season)**
    * Create list of heuristics for V1 of linear fn approximator (optimally play a few more game of Santorini to do this)
    * ~~Create AI class that can interface with existing game code to play randomly~~

* **Week 8**
    * Implement data structure for linear function approximator + means to calculate it given a certain board game state
    * Implement AI class that uses TD learning (should be gradient descent) to play, can also be epsilon greedy.
        * Note: if this is too inefficient I think I will need to go straight to TD(lambda) which uses eligibility traces, or even implement minimax straight away.
    * Test out this basic AI against the random one, see whether win rate makes sense.
    * **If possible**: implement minimax search and incorporate it into RL learning with one of the simpler approaches

* **Week 9**
    * Implement the higher level RL + search approaches and review their relative performance
    * Continue improving the linear function approximator with more features?
* **Week 10**
    * Buffer because things will definitely go wrong somehow
    * Should accomplish all main goals by this week. Final Report is due on 17 April. (End of Week 13)
    * Look into stretch goals, more advanced features to improve the training and playing performance
    * Possibly look into existing research out there detailing more possible techniques.


---
### **The issue of exploration? How do I ensure this takes place...**
---
* Should you train by self-play, or by playing against a random opponent?
    * maybe the opponent can be varied from time-to-time. playing against diff vers of itself, so we don't get stuck in some kind of local equilibrium.
* Can incorporation of some kind of lookahead search help with training using self-play? (scientific papers seem to show that self-play combined with MCTS seems to work)
* incorporate epsilon greedy exploration? + other techniques from multi-armed bandits if really needed
* **Pros**: constantly improve towards the Nash equilibrium
* **Cons**: reach some undesirable stable equilibrium

---
### **Can an AI be trained on a 'knowledge bank' of previous games?**
---
* **Experience replay** is a legitimate strategy for training the value function approximator I think.
* used for DQN, helps to stabilize NNs, since it's an added layer of complexity I think I'll forgo this for now.

---
### **Random AI**
---
Pseudocode for Random AI:
for every turn where it needs to take an action
- retrieve all possible moves (2 workers * 8 moves * 8 building placements = 128 worst case)
- select a random move out of these 128 possible moves and play it

Alternative: (simpler to implement but probably even weaker)
- just keep making random decisions (but could get stuck I think this is a bad idea)

---
### **Linear Value Fn Approximation Heuristics**
---
* There are 2 ways to lose: run out of moves, or opponent gets to level 3
* for convenience, consider a move to be moving + building

**Heuristics to Implement**:

* Tier 1 Importance (initial implementation)
    * options/mobility
        * num of possible moves that opponent's w1 can make + w2
        * num of potential moves your w1 can make, w2 can make
        * 2nd order interactions?
            * your possible num of moves - opponent num of moves
        * number of immediate opportunities to go up one level (0->1, 1->2, 2->3)
        * opportunities for opponent to go up one level
    * height of workers
        * 1 worker on level 1
        * 2 workers on level 1
        * (and so on for level 2)
        * (repeat for opponent's workers)
    * piece distance from centre of board (for you and opponent, each piece)
    * 2nd order interactions (can trim through feature importance later on)

* Tier 2 Importance (see how the earlier features do first, intended to be for more advanced strategies)
    * defensive options (will minimax deal w this?
    * ability to block opponents (defense)
    * ability to build away from opponents without interference (offense)
    * forcing opponent to go down levels (trapping)
