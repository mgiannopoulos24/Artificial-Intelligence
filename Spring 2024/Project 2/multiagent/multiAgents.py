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
        newNumFood = successorGameState.getNumFood()
        currentScore = scoreEvaluationFunction(currentGameState)
        newScore = successorGameState.getScore()

        closestGhostDistance=min([manhattanDistance(newPos,ghost.getPosition())for ghost in newGhostStates])
        
        FoodL= newFood.asList()
        if FoodL:
            closestFoodDistance=min([manhattanDistance(newPos,food) for food in FoodL])
        else:
            closestFoodDistance=0
        

        scoreDifference = newScore - currentScore
        smallScareTime=min(newScaredTimes)
        if smallScareTime!=0:
            closestGhostDistance= closestGhostDistance- (closestGhostDistance*3)
        if action== "Stop":
            return (1/closestFoodDistance)
        else:
            return((15/(closestFoodDistance+1))+(80/(newNumFood+1)))+((closestGhostDistance/8)+scoreDifference)

        

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
            if gameState.isWin() or gameState.isLose() or depth == self.depth * gameState.getNumAgents():
                return self.evaluationFunction(gameState), None
            
            # If the agent is Pac-Man, we maximize; if the agent is a ghost, we minimize.
            if agentIndex == 0:
                return max(
                    (minimax((agentIndex + 1) % gameState.getNumAgents(), depth + 1, gameState.generateSuccessor(agentIndex, action))[0], action)
                    for action in gameState.getLegalActions(agentIndex))
            else:
                return min(
                    (minimax((agentIndex + 1) % gameState.getNumAgents(), depth + 1, gameState.generateSuccessor(agentIndex, action))[0], action)
                    for action in gameState.getLegalActions(agentIndex))
        
        # Start with Pac-Man (agentIndex 0) and depth 0.
        score, action = minimax(0, 0, gameState)
        return action
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        maxResult = float('-inf')
        a = float('-inf')
        b = float('inf')
        for action in actions:
            # MAX (agent index = 0) plays first
            successor = gameState.generateSuccessor(0, action)
            # The first ghost (index = 1) plays next. Depth starts at 0.
            currentResult = self.minValue( successor, 0, 1, a, b )
            if currentResult > maxResult:
                maxResult = currentResult
                maxAction = action
                a = max( (a, currentResult) )
        return maxAction

    def maxValue(self, gameState, currDepth, a, b):
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(0)
        maxValue = float('-inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            # Agent with index == 1 (the first ghost) plays next
            maxValue = max( (maxValue, self.minValue(successor, currDepth, 1, a, b)) )
            if maxValue > b:
                return maxValue
            a = max( (a,maxValue) )
        return maxValue

    def minValue(self, gameState, currDepth, currAgent, a, b):
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(currAgent)
        minValue = float('inf')
        agents = gameState.getNumAgents()
        for action in actions:
            successor = gameState.generateSuccessor(currAgent, action)
            if currAgent < agents - 1:
                # There are still some ghosts to choose their moves, so increase agent index and call minValue again
                minValue = min( (minValue, self.minValue(successor, currDepth, currAgent + 1, a, b)) )
            else:
                # Depth is increased when it is MAX's turn
                minValue = min( (minValue, self.maxValue(successor, currDepth + 1, a, b)) )
            if minValue < a:
                return minValue
            b = min( (b,minValue) )
        return minValue
        util.raiseNotDefined()

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
        def expectimax(gameState, agentIndex, depth=0):
            legalActionList = gameState.getLegalActions(agentIndex)
            numIndex = gameState.getNumAgents() - 1
            bestAction = None
            # If terminal(pos)
            if (gameState.isLose() or gameState.isWin() or depth == self.depth):
                return [self.evaluationFunction(gameState)]
            elif agentIndex == numIndex:
                depth += 1
                childAgentIndex = self.index
            else:
                childAgentIndex = agentIndex + 1

            numAction = len(legalActionList)
            #if player(pos) == MAX: value = -infinity
            if agentIndex == self.index:
                value = -float("inf")
            #if player(pos) == CHANCE: value = 0
            else:
                value = 0

            for legalAction in legalActionList:
                successorGameState = gameState.generateSuccessor(agentIndex, legalAction)
                expectedMax = expectimax(successorGameState, childAgentIndex, depth)[0]
                if agentIndex == self.index:
                    if expectedMax > value:
                        #value, best_move = nxt_val, move
                        value = expectedMax
                        bestAction = legalAction
                else:
                    #value = value + prob(move) * nxt_val
                    value = value + ((1.0/numAction) * expectedMax)
            return value, bestAction

        bestScoreActionPair = expectimax(gameState, self.index)
        bestScore = bestScoreActionPair[0]
        bestMove =  bestScoreActionPair[1]
        return bestMove
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostPositions = currentGameState.getGhostPositions()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    numCapsules = len(currentGameState.getCapsules())
    foodList = currentGameState.getFood().asList()
    numFood = currentGameState.getNumFood()
    badGhost = []
    yummyGhost = []
    total = 0
    win = 0
    lose = 0
    score = 0
    foodScore = 0
    ghost = 0
    if currentGameState.isWin():
        win = 10000000000000000000000000000
    elif currentGameState.isLose():
        lose = -10000000000000000000000000000
    score = 10000 * currentGameState.getScore()
    capsules = 10000000000/(numCapsules+1)
    for food in foodList:
        foodScore += 50/(manhattanDistance(pacmanPosition, food)) * numFood
    for index in range(len(scaredTimes)):
        if scaredTimes[index] == 0:
            badGhost.append(ghostPositions[index])
        else:
            yummyGhost.append(ghostPositions[index])
    for index in range(len(yummyGhost)):
        ghost += 1/(((manhattanDistance(pacmanPosition, yummyGhost[index])) * scaredTimes[index])+1)
    for death in badGhost:
        ghost +=  manhattanDistance(pacmanPosition, death)
    total = win + lose + score + capsules + foodScore + ghost
    return total
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
