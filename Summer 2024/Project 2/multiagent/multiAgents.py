# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Initialize the score
        score = successorGameState.getScore()

        # Calculate the distance to the closest food
        foodList = newFood.asList()
        if foodList:
            minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])
            # Closer food gives a higher score
            score += 10 / (minFoodDistance + 1)  # Add small weight for closer food
        
        # Calculate the distance to the closest food
        foodList = newFood.asList()
        if foodList:
            minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])
            # Closer food gives a higher score
            score += 10 / (minFoodDistance + 1)  # Add small weight for closer food

        # Calculate the distance to each ghost
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            distanceToGhost = manhattanDistance(newPos, ghostPos)
            if newScaredTimes[i] > 0:  # Ghost is scared
                # Closer scared ghosts give a higher score
                score += 200 / (distanceToGhost + 1)
            else:  # Ghost is not scared
                if distanceToGhost < 2:
                    # Penalize heavily if ghost is too close
                    score -= 1000

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, depth, gameState):
            # Check if game has ended (win/loss) or reached max depth
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's turn (Maximizer)
            if agentIndex == 0:
                return max(minimax(1, depth, gameState.generateSuccessor(agentIndex, action))
                           for action in gameState.getLegalActions(agentIndex))

            # Ghosts' turn (Minimizer)
            else:
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()  # cycle through agents
                nextDepth = depth + 1 if nextAgent == 0 else depth

                return min(minimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action))
                           for action in gameState.getLegalActions(agentIndex))

        # Pacman starts at agentIndex = 0
        # Get the best action based on minimax
        legalActions = gameState.getLegalActions(0)
        scores = [minimax(1, 0, gameState.generateSuccessor(0, action)) for action in legalActions]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        
        # Pick one of the best actions
        return legalActions[bestIndices[0]]
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            # If game is over (win/lose), or depth is reached, return the evaluation function value
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's turn (maximize)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState, alpha, beta)
            
            # Ghosts' turn (minimize)
            else:
                return minValue(agentIndex, depth, gameState, alpha, beta)

        def maxValue(agentIndex, depth, gameState, alpha, beta):
            actions = gameState.getLegalActions(agentIndex)
            if not actions:  # No legal actions
                return self.evaluationFunction(gameState)

            v = float('-inf')
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, alphaBeta(1, depth, successor, alpha, beta))  # Next agent is ghost 1
                if v > beta:
                    return v  # Beta pruning
                alpha = max(alpha, v)
            return v

        def minValue(agentIndex, depth, gameState, alpha, beta):
            actions = gameState.getLegalActions(agentIndex)
            if not actions:  # No legal actions
                return self.evaluationFunction(gameState)

            v = float('inf')
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()  # Rotate between Pacman and ghosts
            if nextAgent == 0:  # Back to Pacman's turn, increase depth
                nextDepth = depth + 1
            else:
                nextDepth = depth

            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, alphaBeta(nextAgent, nextDepth, successor, alpha, beta))
                if v < alpha:
                    return v  # Alpha pruning
                beta = min(beta, v)
            return v

        # Start with Pacman (agentIndex=0) and initial alpha, beta values
        alpha = float('-inf')
        beta = float('inf')
        
        # Get Pacman's legal actions
        actions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float('-inf')

        # Perform alpha-beta pruning on Pacman's choices
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = alphaBeta(1, 0, successor, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, value)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agentIndex, depth, gameState):
            # If game is over (win/lose), or max depth is reached, return the evaluation function value
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's turn (maximizing)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)

            # Ghost's turn (expectation of the outcomes)
            else:
                return expValue(agentIndex, depth, gameState)

        def maxValue(agentIndex, depth, gameState):
            actions = gameState.getLegalActions(agentIndex)
            if not actions:  # No legal actions
                return self.evaluationFunction(gameState)

            # Maximizing value for Pacman
            bestValue = float('-inf')
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                bestValue = max(bestValue, expectimax(1, depth, successor))  # Next agent is ghost 1
            return bestValue

        def expValue(agentIndex, depth, gameState):
            actions = gameState.getLegalActions(agentIndex)
            if not actions:  # No legal actions
                return self.evaluationFunction(gameState)

            # Calculate expected value for the ghosts
            totalValue = 0
            prob = 1.0 / len(actions)  # Uniform probability for each action
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            if nextAgent == 0:  # Pacman's turn, increase depth
                nextDepth = depth + 1
            else:
                nextDepth = depth

            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                totalValue += prob * expectimax(nextAgent, nextDepth, successor)
            return totalValue

        # Start with Pacman (agentIndex=0) and perform expectimax
        actions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float('-inf')

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(1, 0, successor)
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Get useful information from the current game state
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()

    # Start with the base score of the current game state
    score = currentGameState.getScore()

    # Evaluate food distance: Encourage Pacman to move toward the closest food
    foodList = food.asList()
    if foodList:
        closestFoodDistance = min([manhattanDistance(pacmanPos, foodPos) for foodPos in foodList])
        score += 10.0 / closestFoodDistance  # Incentivize moving towards food
        score -= 3.0 * len(foodList)  # Penalize for more remaining food

    # Evaluate capsule distance: Encourage Pacman to move toward the closest capsule if ghosts are nearby
    if capsules:
        closestCapsuleDistance = min([manhattanDistance(pacmanPos, capsulePos) for capsulePos in capsules])
        # Lower capsule bonus slightly, but still prioritize them with ghosts around
        if any([ghostState.scaredTimer == 0 for ghostState in ghostStates]):  # Only incentivize capsules if ghosts are not scared
            score += 15.0 / (closestCapsuleDistance + 1)  # Incentivize moving toward capsules if ghosts are a threat
        score += 50.0 * len(capsules)  # Slight bonus for remaining capsules

    # Evaluate ghost proximity: Avoid non-scared ghosts, move towards scared ghosts
    for i, ghostState in enumerate(ghostStates):
        ghostPos = ghostState.getPosition()
        ghostDistance = manhattanDistance(pacmanPos, ghostPos)

        if scaredTimes[i] > 0:  # Ghost is scared
            # Diminishing return as Pacman gets closer to a scared ghost
            score += 200.0 / (ghostDistance + 1) if ghostDistance > 1 else 100.0  # Reduced bonus if already very close
        else:  # Ghost is dangerous
            if ghostDistance < 2:
                score -= 1200.0  # Very heavy penalty if too close
            else:
                score -= 150.0 / (ghostDistance + 1)  # Penalize proportionally to ghost proximity

    # Add large bonuses for win state, large penalties for loss state
    if currentGameState.isWin():
        score += 5000.0
    if currentGameState.isLose():
        score -= 5000.0

    return score

# Abbreviation
better = betterEvaluationFunction
