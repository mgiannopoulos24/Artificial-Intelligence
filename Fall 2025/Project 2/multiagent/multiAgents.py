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
        # Initialize score with the current game state score
        score = successorGameState.getScore()

        # Distance to closest food
        foodList = newFood.asList()  # Get the list of food positions
        if foodList:
            minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])
            # The closer Pacman is to the food, the higher the score
            score += 10.0 / minFoodDistance

        # Ghost proximity handling
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            distanceToGhost = manhattanDistance(newPos, ghostPos)

            if scaredTime > 0:
                # Ghost is scared, prioritize eating it
                score += 10.0 / distanceToGhost
            else:
                # Ghost is not scared, avoid getting too close
                if distanceToGhost > 0:  # Avoid division by zero
                    score -= 10.0 / distanceToGhost

        # Avoid stopping unless it's the only legal action
        if action == Directions.STOP:
            score -= 50  # Penalize stopping

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
            # Base case: Check if the game is over or if we've reached the maximum depth
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman (maximizer) is agentIndex 0
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            # Ghosts (minimizer) are agentIndex 1 or higher
            else:
                return minValue(agentIndex, depth, gameState)
        
        def maxValue(agentIndex, depth, gameState):
            # Initialize max value
            v = float('-inf')
            # Get Pacman's legal actions
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            # Iterate through all possible actions and find the maximum value
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, minimax(1, depth, successor))  # Ghosts start at index 1
            return v

        def minValue(agentIndex, depth, gameState):
            # Initialize min value
            v = float('inf')
            # Get the current agent's legal actions (ghosts)
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            # Get the next agent's index and check if we need to increase depth
            nextAgent = agentIndex + 1
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0  # Go back to Pacman
                depth += 1  # Increase the depth since we've gone through all agents

            # Iterate through all possible actions and find the minimum value
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, minimax(nextAgent, depth, successor))
            return v

        # Pacman (agentIndex 0) will choose the action with the best minimax score
        bestAction = None
        bestScore = float('-inf')

        for action in gameState.getLegalActions(0):  # Pacman's legal actions
            successor = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, successor)  # Start with Ghost 1, depth 0
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(agentIndex, depth, gameState, alpha, beta):
            # Base case: Check if the game is over or if we've reached the maximum depth
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman (maximizer) is agentIndex 0
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState, alpha, beta)
            # Ghosts (minimizer) are agentIndex 1 or higher
            else:
                return minValue(agentIndex, depth, gameState, alpha, beta)

        def maxValue(agentIndex, depth, gameState, alpha, beta):
            # Initialize max value
            v = float('-inf')
            # Get Pacman's legal actions
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            # Iterate through all possible actions and update alpha-beta values
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, alphabeta(1, depth, successor, alpha, beta))  # Ghosts start at index 1
                if v > beta:
                    return v  # Prune the remaining branches
                alpha = max(alpha, v)
            return v

        def minValue(agentIndex, depth, gameState, alpha, beta):
            # Initialize min value
            v = float('inf')
            # Get the current agent's legal actions (ghosts)
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            # Get the next agent's index and check if we need to increase depth
            nextAgent = agentIndex + 1
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0  # Go back to Pacman
                depth += 1  # Increase the depth since we've gone through all agents

            # Iterate through all possible actions and update alpha-beta values
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, alphabeta(nextAgent, depth, successor, alpha, beta))
                if v < alpha:
                    return v  # Prune the remaining branches
                beta = min(beta, v)
            return v

        # Pacman (agentIndex 0) will choose the action with the best alpha-beta score
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):  # Pacman's legal actions
            successor = gameState.generateSuccessor(0, action)
            score = alphabeta(1, 0, successor, alpha, beta)  # Start with Ghost 1, depth 0
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, score)

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
        # Call the expectimax function starting from Pacman (agent index 0)
        return self.expectimax(gameState, 0, 0)

    def expectimax(self, gameState, agentIndex, currentDepth):
        # If at terminal state or max depth, return evaluation
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Determine whether the agent is Pacman (agentIndex 0) or a ghost
        if agentIndex == 0:  # Pacman: Maximize utility
            return self.maxValue(gameState, agentIndex, currentDepth)
        else:  # Ghost: Compute expected utility
            return self.expValue(gameState, agentIndex, currentDepth)

    def maxValue(self, gameState, agentIndex, currentDepth):
        # Initialize variables
        legalMoves = gameState.getLegalActions(agentIndex)
        if not legalMoves:
            return self.evaluationFunction(gameState)

        # Get the maximum value over all legal moves for Pacman
        maxScore = float('-inf')
        bestAction = None
        for action in legalMoves:
            successorState = gameState.generateSuccessor(agentIndex, action)
            score = self.expectimax(successorState, 1, currentDepth)  # Next agent will be the first ghost
            if score > maxScore:
                maxScore = score
                bestAction = action

        if currentDepth == 0:
            return bestAction 
        else:
            return maxScore  

    def expValue(self, gameState, agentIndex, currentDepth):
        legalMoves = gameState.getLegalActions(agentIndex)
        if not legalMoves:
            return self.evaluationFunction(gameState)

        # Compute the expected value for the ghost moving randomly
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgent == 0:  # If next agent is Pacman, increment depth
            currentDepth += 1

        totalScore = 0
        probability = 1.0 / len(legalMoves)  # Assume uniform random move selection
        for action in legalMoves:
            successorState = gameState.generateSuccessor(agentIndex, action)
            score = self.expectimax(successorState, nextAgent, currentDepth)
            totalScore += score * probability

        return totalScore

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    score = currentGameState.getScore()
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    ghostScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    evaluation = score

    # Factor 1: Distance to the closest food (maximize proximity to food)
    if food:
        minFoodDistance = min(manhattanDistance(pacmanPos, foodPos) for foodPos in food)
        evaluation += 1.0 / (minFoodDistance + 1)  

    # Factor 2: Ghost proximity (avoid unless scared)
    for ghostState, scaredTime in zip(ghostStates, ghostScaredTimes):
        ghostPos = ghostState.getPosition()
        ghostDistance = manhattanDistance(pacmanPos, ghostPos)

        if scaredTime > 0:
            # If ghosts are scared, get closer to them
            evaluation += 200 / (ghostDistance + 1)  
        else:
            # Otherwise, avoid the ghosts
            if ghostDistance <= 2:
                evaluation -= 100  

    # Factor 3: Number of remaining food pellets (prefer fewer food pellets left)
    evaluation -= 10 * len(food)  

    # Factor 4: Capsules (prefer being near capsules)
    capsules = currentGameState.getCapsules()
    if capsules:
        minCapsuleDistance = min(manhattanDistance(pacmanPos, capsulePos) for capsulePos in capsules)
        evaluation += 10 / (minCapsuleDistance + 1)  

    return evaluation

# Abbreviation
better = betterEvaluationFunction
