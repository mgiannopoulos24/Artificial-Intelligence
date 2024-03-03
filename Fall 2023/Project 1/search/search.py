# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    currPath = []           # The path that is popped from the frontier in each loop
    currState =  problem.getStartState()

    if problem.isGoalState(currState):     # Checking if the start state is also a goal state
        return currPath
    frontier = util.Stack()
    frontier.push( (currState, currPath) )     # Insert just the start state, in order to pop it first
    explored = set()
    while not frontier.isEmpty():
        currState, currPath = frontier.pop()    # Popping a state and the corresponding path
        # To pass autograder.py question1:
        if problem.isGoalState(currState):
            return currPath
        explored.add(currState)
        for s in problem.getSuccessors(currState):
            if s[0] not in explored:
                # Lecture code:
                # if problem.isGoalState(s[0]):
                #     return currPath + [s[1]]
                frontier.push( (s[0], currPath + [s[1]]) )      # Adding the successor and its path to the frontier

    return []       # If this point is reached, a solution could not be found.

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    currPath = []           # The path that is popped from the frontier in each loop
    currState =  problem.getStartState()    # The state(position) that is popped for the frontier in each loop

    if problem.isGoalState(currState):     # Checking if the start state is also a goal state
        return currPath

    frontier = util.Queue()
    frontier.push( (currState, currPath) )     # Insert just the start state, in order to pop it first
    explored = set()
    while not frontier.isEmpty():
        currState, currPath = frontier.pop()    
        if problem.isGoalState(currState):
            return currPath
        explored.add(currState)
        frontierStates = [ t[0] for t in frontier.list ]
        for s in problem.getSuccessors(currState):
            if s[0] not in explored and s[0] not in frontierStates:
                frontier.push( (s[0], currPath + [s[1]]) )     

    return []       # If this point is reached, a solution could not be found


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    currPath = []           # The path that is popped from the frontier in each loop
    currState = problem.getStartState()     # The state(position) that is popped for the frontier in each loop
    frontier = util.PriorityQueue()
    frontier.push((currState, currPath), 0)
    explored = set()

    while not frontier.isEmpty():
        currState, currPath = frontier.pop()
        if problem.isGoalState(currState):
            return currPath
        explored.add(currState)
        frontierStates = [ i[2][0] for i in frontier.heap ]     # frontier.heap[i][2] is the state tuple: (position, path)
        for s in problem.getSuccessors(currState):
            successorPath = currPath + [s[1]]        # The path to the new successor
            if s[0] not in explored and s[0] not in frontierStates:
                frontier.push( (s[0], successorPath), problem.getCostOfActions(successorPath) )
            else:
                # The same state already exists
                for i in range(0, len(frontierStates)):
                    # Finding it
                    if s[0] == frontierStates[i]:
                        # The stored path and the new path costs have to be compared
                        updatedCost = problem.getCostOfActions(successorPath)
                        storedCost = frontier.heap[i][0]    # frontier.heap[i] is a tuple: (cost, counter, (node, path))
                        if storedCost > updatedCost:
                            # The cost must be updated
                            # Plus, (s[0], <stored_path>) must be changed to (s[0], successorPath)
                            # Tuples are immutable, so frontier.heap[i] must be reconstructed
                            # First we manually change just the path, while keeping the cost unchanged
                            frontier.heap[i] = (storedCost, frontier.heap[i][1] , (s[0], successorPath) )
                            # and then we update the cost
                            frontier.update( (s[0], successorPath), updatedCost )

    return []       # If this point is reached, a solution could not be found.

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    have_visited = set()

    start_state = problem.getStartState()
    frontier.push((start_state, list()), heuristic(start_state, problem))

    while(not frontier.isEmpty()):
        state, path = frontier.pop()  # Take current state and path
        have_visited.add(state)  # Visit the current state

        # Found goal state
        if (problem.isGoalState(state)): return path

        successors = problem.getSuccessors(state)
        for successor in successors:

            # Already have visited the particular node
            if (successor[0] in have_visited): continue
            
            # Search to see if successor already exists in the frontier(PQ)
            frontier_exists = False
            for element in frontier.heap:
                if (successor[0] == element[2][0]):
                    frontier_exists = True
                    break
            
            heuristic_eval = heuristic(successor[0], problem)  # heuristic cost

            new_path = path + [successor[1]]
            new_priority = problem.getCostOfActions(new_path)

            # State does not exist either in searched state nor in the frontier, insert it
            if (not frontier_exists):
                frontier.push((successor[0], new_path), new_priority + heuristic_eval)
            
            # Successor exists in the frontier with a higher path cost - update its path cost
            elif (problem.getCostOfActions(element[2][1]) > new_priority):
                frontier.update((successor[0], new_path), new_priority + heuristic_eval)
    
    # Solution/Path does not exist
    return None
    


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
