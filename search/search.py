# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

from util import *
#from math import inf
#import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def search(problem, fringe):
    initial_state = problem.getStartState()
    initial_actions = []
    initial_candidate = (initial_state, initial_actions)
    fringe.push(initial_candidate)
    closed_set = set()
    while not fringe.isEmpty():
        candidate = fringe.pop()
        state, actions = candidate
        if problem.isGoalState(state):
            return actions
        if state not in closed_set:
            closed_set.add(state)
            candidate_successors = problem.getSuccessors(state)
            candidate_successors = filter(lambda x: x[0] not in closed_set, candidate_successors)
            candidate_successors = map(lambda x: (x[0], actions + [x[1]]), candidate_successors)
            for candidate in candidate_successors:
                fringe.push(candidate)

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    fringe = Stack()
    prev_states = set()
    initial_candidate = (problem.getStartState(), []) #(state[Coord], action[N,E,S,W])
    fringe.push(initial_candidate)
        
    while not fringe.isEmpty():
        state, actions  = fringe.pop()
        if problem.isGoalState(state):
            return actions
        if state not in prev_states:
            prev_states.add(state)
            for suc in filter(lambda x: x[0] not in prev_states, problem.getSuccessors(state)):
                fringe.push((suc[0], actions + [suc[1]]))
        
    return []


def printFringe(fringe):
    print "Fringe: ", fringe.heap[0:]


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """

    fringe = Queue()
    prev_states = set()
    initial_candidate = (problem.getStartState(), []) #(state[Coord], action[N,E,S,W])
    fringe.push(initial_candidate)
        
    while not fringe.isEmpty():
        state, actions = fringe.pop()
        if problem.isGoalState(state):
            return actions
        if state not in prev_states:
            prev_states.add(state)
            for suc in filter(lambda x: x[0] not in prev_states, problem.getSuccessors(state)):
                fringe.push((suc[0], actions + [suc[1]]))
                
    return []


def uniformCostSearch(problem):
    "Search the node of least total cost first."
    
    gameState = problem.getGameState()
    problem.setCostFn(lambda s: cost(s, problem))
    fringe = PriorityQueue()
    prev_states_priorities = set()
    initial_candidate = (problem.getStartState(), []) #(state[Coord], action[N,E,S,W])
    fringe.push(initial_candidate, 0)
        
    while not fringe.isEmpty():
        state, actions = fringe.pop()
        priority = problem.getCostOfActions(actions)
        if problem.isGoalState(state):
            print "Priority: ", priority
            return actions
        if (state, priority) not in prev_states_priorities:
            prev_states_priorities.add((state, priority))
            prev_s, prev_p = zip(*prev_states_priorities)
            for suc in problem.getSuccessors(state):
                new_actions = actions + [suc[1]]
                new_priority = problem.getCostOfActions(new_actions)
                if (suc[0] not in prev_s) or (prev_p[prev_s.index(suc[0])] > new_priority): 
                    fringe.push((suc[0], new_actions), new_priority)
                
    return []

def cost(state, problem):
    gameState = problem.getGameState()
    min_ghost_distance = 5
    cost = 1
    if gameState.hasFood(state[0], state[1]):
        cost = 0
    ghost_distances = map(lambda p: manhattanDistance(p, state), gameState.getGhostPositions())
    if filter(lambda d: d <= min_ghost_distance, ghost_distances):
        cost = 10
    return cost

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    
    #
    myInf = 1000000000
    #
    
    closedSet = set() #nodes already evaluated
    openSet = set() #currently discovered nodes, still to be evaluated
    initial_state = problem.getStartState()
    openSet.add(initial_state)
    
    cameFrom = dict() #for each node, the most efficient previous step

    gScore = dict() #the cost of getting from the start node to the current node
    gScore[initial_state] = 0 
    
    fScore = PriorityQueue() #the total cost of getting from the start node to the goal by passing by the current node (partly known, partly heuristic)
    fScore.push(initial_state, heuristic(initial_state, problem)) #for the first node, that value is completely heuristic

    while openSet:
        current_state = fScore.pop()
        if problem.isGoalState(current_state):
            return reconstruct_actions(cameFrom, current_state)
        openSet.remove(current_state)
        closedSet.add(current_state)
        for suc in problem.getSuccessors(current_state):
            suc_state = suc[0]
            suc_action = suc[1]
            suc_cost = suc[2]
            if suc_state in closedSet: #ignore the successors which are already evaluated
                continue
            tentative_gScore = gScore[current_state] + suc_cost #the distance from start to a node
            if suc not in openSet: #discover a new node
                openSet.add(suc_state)
                gScore[suc_state] = myInf
                fScore.push(suc_state, myInf)
            elif tentative_gScore >= gScore[suc_state]:
                continue #this is not a better path

            #this path is the best until now
            cameFrom[suc_state] = (current_state, suc_action)
            gScore[suc_state] = tentative_gScore
            fScore.push(suc_state, gScore[suc_state] + heuristic(suc_state, problem))

    return []

def reconstruct_actions(cameFrom, current_state):
    actions = []
    while current_state in cameFrom:
        current_state, act = cameFrom[current_state]
        actions = [act] + actions
    return actions
    

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
