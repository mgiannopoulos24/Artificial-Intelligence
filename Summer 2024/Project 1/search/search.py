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
    # Initialize a stack for the frontier of nodes to explore
    frontier = util.Stack()
    
    # Push the starting state along with an empty path and a cost of 0
    frontier.push((problem.getStartState(), [], 0))
    
    # Set of visited nodes to avoid revisiting
    visited = set()
    
    while not frontier.isEmpty():
        # Pop the current state, path, and cost
        state, path, cost = frontier.pop()
        
        # If the state has been visited, skip it
        if state in visited:
            continue
        
        # Mark this state as visited
        visited.add(state)
        
        # If the state is the goal, return the path
        if problem.isGoalState(state):
            return path
        
        # Expand the current state by getting its successors
        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in visited:
                # Add the successor to the frontier with the updated path and cost
                frontier.push((successor, path + [action], cost + stepCost))
    
    # If no solution is found, return an empty path
    return []

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Initialize a queue for the frontier of nodes to explore
    frontier = util.Queue()
    
    # Push the starting state along with an empty path and a cost of 0
    frontier.push((problem.getStartState(), [], 0))
    
    # Set of visited nodes to avoid revisiting
    visited = set()
    
    while not frontier.isEmpty():
        # Pop the current state, path, and cost
        state, path, cost = frontier.pop()
        
        # If the state has been visited, skip it
        if state in visited:
            continue
        
        # Mark this state as visited
        visited.add(state)
        
        # If the state is the goal, return the path
        if problem.isGoalState(state):
            return path
        
        # Expand the current state by getting its successors
        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in visited:
                # Add the successor to the frontier with the updated path and cost
                frontier.push((successor, path + [action], cost + stepCost))
    
    # If no solution is found, return an empty path
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Initialize a priority queue for the frontier of nodes to explore
    frontier = util.PriorityQueue()
    
    # Push the starting state with a cost of 0
    frontier.push((problem.getStartState(), [], 0), 0)
    
    # Dictionary to store the lowest cost to reach each state
    visited = {}
    
    while not frontier.isEmpty():
        # Pop the current state, path, and cost with the lowest cost from the priority queue
        state, path, cost = frontier.pop()
        
        # If the state has been visited with a lower cost, skip it
        if state in visited and visited[state] <= cost:
            continue
        
        # Mark this state as visited with its cost
        visited[state] = cost
        
        # If the state is the goal, return the path
        if problem.isGoalState(state):
            return path
        
        # Expand the current state by getting its successors
        for successor, action, stepCost in problem.getSuccessors(state):
            new_cost = cost + stepCost
            if successor not in visited or new_cost < visited[successor]:
                # Add the successor to the frontier with the updated path and cost
                frontier.push((successor, path + [action], new_cost), new_cost)
    
    # If no solution is found, return an empty path
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Initialize a priority queue for the frontier of nodes to explore
    frontier = util.PriorityQueue()
    
    # Push the starting state with a cost of 0 and the heuristic estimate
    start_state = problem.getStartState()
    frontier.push((start_state, [], 0), heuristic(start_state, problem))
    
    # Dictionary to store the lowest cost to reach each state
    visited = {}
    
    while not frontier.isEmpty():
        # Pop the current state, path, and cost with the lowest combined cost + heuristic
        state, path, cost = frontier.pop()
        
        # If the state has been visited with a lower cost, skip it
        if state in visited and visited[state] <= cost:
            continue
        
        # Mark this state as visited with its cost
        visited[state] = cost
        
        # If the state is the goal, return the path
        if problem.isGoalState(state):
            return path
        
        # Expand the current state by getting its successors
        for successor, action, stepCost in problem.getSuccessors(state):
            new_cost = cost + stepCost
            if successor not in visited or new_cost < visited[successor]:
                # Add the successor to the frontier with the updated path, cost, and heuristic
                priority = new_cost + heuristic(successor, problem)
                frontier.push((successor, path + [action], new_cost), priority)
    
    # If no solution is found, return an empty path
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
