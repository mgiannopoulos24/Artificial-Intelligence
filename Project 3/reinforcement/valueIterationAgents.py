# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            newValues = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    # Terminal states have a value of 0
                    newValues[state] = 0
                else:
                    # Find the maximum Q-value over all possible actions
                    maxQ = float("-inf")
                    for action in self.mdp.getPossibleActions(state):
                        qValue = self.computeQValueFromValues(state, action)
                        if qValue > maxQ:
                            maxQ = qValue
                    newValues[state] = maxQ
            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        qValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            value = self.values[nextState]
            qValue += prob * (reward + self.discount * value)
        return qValue
    

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        legalActions = self.mdp.getPossibleActions(state)

        if not legalActions:
            return None

        # initialize the best action to the first legal action
        bestAction = legalActions[0]
        bestActionValue = self.computeQValueFromValues(state, bestAction)

        # loop over all legal actions to find the one with the highest Q-value
        for action in legalActions:
            actionValue = self.computeQValueFromValues(state, action)
            if actionValue > bestActionValue:
                bestAction = action
                bestActionValue = actionValue

        return bestAction
       

        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()
        for i in range(self.iterations):
            state = random.choice(states)
            if not self.mdp.isTerminal(state):
                maxQ = float("-inf")
                for action in self.mdp.getPossibleActions(state):
                    qVal = self.computeQValueFromValues(state, action)
                    if qVal > maxQ:
                        maxQ = qVal
                self.values[state] = maxQ

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                    if nextState in predecessors:
                        predecessors[nextState].add(state)
                    else:
                        predecessors[nextState] = {state}

        priorityQueue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                qValues = [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]
                diff = abs(self.values[state] - max(qValues))
                priorityQueue.push(state, -diff)

        for i in range(self.iterations):
            if priorityQueue.isEmpty():
                break

            state = priorityQueue.pop()
            if not self.mdp.isTerminal(state):
                qValues = [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]
                self.values[state] = max(qValues)

            for p in predecessors[state]:
                if not self.mdp.isTerminal(p):
                    qValues = [self.computeQValueFromValues(p, action) for action in self.mdp.getPossibleActions(p)]
                    diff = abs(self.values[p] - max(qValues))
                    if diff > self.theta:
                        priorityQueue.update(p, -diff)

