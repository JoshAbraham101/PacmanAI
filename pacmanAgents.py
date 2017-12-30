# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent
from heuristics import *
import random
import math

lastAction = None
capsules = None
chaseStepCount = 30
capsuleHit = False
ghostEaten = 0

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];

class GreedyAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(scoreEvaluation(state), action) for state, action in successors]
        # get best choice
        bestScore = max(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)


class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write BFS Algorithm instead of returning Directions.STOP
        fringeSuccessors = []                       # List of all fringe (childless) nodes, except win state nodes
        winSuccessors = []                          # List of all win (goal) nodes

        legal = state.getLegalPacmanActions()
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # Checking the first expanded nodes for None conditions.
        # None condition here will only be true if function call limit value is reduced.
        for item in successors:
            firstChildNode, firstChildDirection = item
            if firstChildNode is None or firstChildNode.isLose():
                successors.remove(item)               # if Lose/None state is reached, remove it from solution list

        while successors:                             # BFS loop
            newState, direction = successors[0]       # Expand and pop FIRST element in BFS (FIFO)
            if newState.isWin() is True:
                successors.pop(0)                     # if win state is reached, add state to WIN list
                winSuccessors.append((newState, direction))
            elif newState.isLose() is False:
                successors.pop(0)                     # Pop FIRST element in BFS (FIFO) when it has successors
                newLegal = newState.getLegalPacmanActions()
                for newAction in newLegal:
                    successorCheck = newState.generatePacmanSuccessor(newAction)
                    if successorCheck is None:        # if call limit (none state) is reached, add parent to FRINGE list
                        fringeSuccessors.append((newState, direction))
                        continue                      # Break for loop to prevent repeated addition of same fringe node
                    successors.append((successorCheck, direction))
            else:
                successors.pop(0)                     # if Lose state occurs, remove it from possible solutions

        # Return action leading to shallowest fringe node with best score
        # First check WIN states for maximum score
        if winSuccessors:
            scored = [(scoreEvaluation(winState), winAction) for winState, winAction in winSuccessors]
            bestScore = max(scored)[0]
            bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
            return bestActions[0]
        # if WIN is empty, check remaining FRINGE states for maximum score
        else:
            scored = [(scoreEvaluation(fringeState), fringeAction) for fringeState, fringeAction in fringeSuccessors]
            bestScore = max(scored)[0]
            bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
            return bestActions[0]

class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write DFS Algorithm instead of returning Directions.STOP
        fringeSuccessors = []  # List of all fringe (childless) nodes, except win state nodes
        winSuccessors = []  # List of all win (goal) nodes

        legal = state.getLegalPacmanActions()
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # Checking the first expanded nodes for None conditions.
        # None condition here will only be true if function call limit value is reduced.
        for item in successors:
            firstChildNode, firstChildDirection = item
            if firstChildNode is None or firstChildNode.isLose():
                successors.remove(item)  # if Lose/None state is reached, remove it from solution list

        while successors:                           # DFS loop
            newState, direction = successors[-1]    # Expand and pop LAST element in DFS (FIFO)
            if newState.isWin() is True:
                successors.pop()                    # if win state is reached, add state to WIN list
                winSuccessors.append((newState, direction))
            elif newState.isLose() is False:
                successors.pop()                    # Pop LAST element in DFS (LIFO) when it has successors
                newLegal = newState.getLegalPacmanActions()
                for newAction in newLegal:
                    successorCheck = newState.generatePacmanSuccessor(newAction)
                    if successorCheck is None:      # if call limit (none state) is reached, add parent to FRINGE list
                        fringeSuccessors.append((newState, direction))
                        continue                    # Break for loop to prevent repeated addition of same fringe node
                    successors.append((successorCheck, direction))
            else:
                successors.pop()                    # if Lose state occurs, remove it from possible solutions

        # Return action leading to shallowest fringe node with best score
        # First check WIN states for maximum score
        if winSuccessors:
            scored = [(scoreEvaluation(winState), winAction) for winState, winAction in winSuccessors]
            bestScore = max(scored)[0]
            bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
            return bestActions[0]
        # if WIN is empty, check remaining FRINGE states for maximum score
        else:
            scored = [(scoreEvaluation(fringeState), fringeAction) for fringeState, fringeAction in fringeSuccessors]
            bestScore = max(scored)[0]
            bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
            return bestActions[0]

class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write A* Algorithm instead of returning Directions.STOP
        countReached = False                               # Flag to check if call limit is reached
        outputSuccessors = []                              # Final successor list with fringe nodes
        direction = Directions.STOP
        depth = 1
        legal = state.getLegalPacmanActions()
        successors = [(state.generatePacmanSuccessor(action), action, depth) for action in legal]
        while (successors):                                     # A* Loop
            for i in range(len(successors)):
                if successors[i][0] is not None:
                    childNode, childDirection, childDepth = successors[i]
                    if childNode.isWin() is True:
                        return childDirection                   # if goal state is reached return direction (action)
                    elif childNode.isLose() is True:
                        continue                                # if lose state, do not add to outputSuccessor list
                    # Add remaining states to outputSuccessor list with their path cost
                    pathCost = childDepth - (scoreEvaluation(childNode) - scoreEvaluation(state))
                    outputSuccessors.append((childNode, childDirection, childDepth, pathCost))
                else:
                    countReached = True

            # Sort outputSuccessor list to find successor states of node with minimum path cost and pop it from list
            if countReached is False:
                outputSuccessors = sorted(outputSuccessors, key=lambda cost: cost[3])
                if not outputSuccessors or outputSuccessors is None:
                    return direction
                minState, direction, minNodeDepth, minNodeCost = outputSuccessors[0]
                legal = minState.getLegalPacmanActions()
                outputSuccessors.pop(0)
                depth = minNodeDepth + 1
                successors = [(minState.generatePacmanSuccessor(action), direction, depth) for action in legal]
            else:
                successors = []

        # Return action leading to shallowest fringe node with best score
        scored = [(scoreEvaluation(fringeState), fringeAction, fringeDepth) for fringeState, fringeAction, fringeDepth,
                                                                                fringeCost in outputSuccessors]
        if not outputSuccessors or outputSuccessors is None:
            return direction
        bestScore = max(scored)[0]
        bestActions = [(pair[1], pair[2]) for pair in scored if pair[0] == bestScore]
        leastDepth = min(bestActions)[1]
        finalActions = [pair[0] for pair in bestActions if pair[1] == leastDepth]
        return finalActions[0]



class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0, 5):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP
        currentScore = scoreEvaluation(state)
        bestSeqScore = -20000
        noneReached = False
        possible = state.getAllPossibleActions()

        ##Generate first action sequence
        for i in range(0, len(self.actionList)):
            self.actionList[i] = possible[random.randint(0, len(possible) - 1)]
        tempState = state
        ##Score Evaluation of first action sequence
        for i in range(0, len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                prevState = tempState
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
                if tempState is None:
                    noneReached = True
                    tempState = prevState
                    break
            else:
                break;
        bestSeqScore = scoreEvaluation(tempState)

        ## generate random action in sequence and check for better score
        while noneReached is False:
            newSeqScore = 0
            newSequence = self.actionList[:]
            for i in range(0, len(newSequence)):
                if random.randint(0, 1) == 1:       ## 50% chance of random action
                    newSequence[i] = possible[random.randint(0, len(possible) - 1)]
            tempState = state;
            for i in range(0, len(newSequence)):
                if tempState.isWin() + tempState.isLose() == 0:
                    prevState = tempState
                    tempState = tempState.generatePacmanSuccessor(newSequence[i]);
                    if tempState is None:
                        noneReached = True
                        tempState = prevState
                        break
                else:
                    break
            ##tempState = state only if None is returned from the first action of the sequence
            if tempState is not state:
                newSeqScore = scoreEvaluation(tempState)
            if newSeqScore >= bestSeqScore:
                bestSeqScore = newSeqScore
                self.actionList = newSequence[:]

        return self.actionList[0]

class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0, 5):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        newSeqScore = 0
        bestScore = ("",-20000)
        noneReached = False
        population = []
        possible = state.getAllPossibleActions()

        ##initial generation of population
        for i in range(0,8):
            for j in range(0, len(self.actionList)):
                self.actionList[j] = possible[random.randint(0, len(possible) - 1)]
            population.append(self.actionList[:])

        while noneReached is False:
            rankedPopulation = []
            for i in range(0, len(population)):
                tempState = state
                for j in range(0, len(self.actionList)):
                    if tempState.isWin() + tempState.isLose() == 0:
                        prevState = tempState
                        tempState = tempState.generatePacmanSuccessor(population[i][j]);
                        if tempState is None:
                            noneReached = True
                            tempState = prevState
                            break
                    else:
                        break
                ##tempState = state only if None is returned from the first action of the sequence
                if tempState is not state:
                    newSeqScore = scoreEvaluation(tempState)
                    rankedPopulation.append((population[i][:],newSeqScore))
                else:
                    rankedPopulation.append((population[i][:], 0))

            rankedPopulation = sorted(rankedPopulation, key=lambda score: score[1])
            highestPopulationScore = rankedPopulation[len(rankedPopulation)-1][1]
            if highestPopulationScore >= bestScore[1]:
                bestScore = (rankedPopulation[len(rankedPopulation)-1][0][0], highestPopulationScore)

            ## Selecting pair using rank selection
            newPopulation = []
            for i in range(0,4):
                value1 = random.randint(1, 36)
                value2 = random.randint(1, 36)
                rankSelected1 = self.rankSelection(value1)
                rankSelected2 = self.rankSelection(value2)
                selectedParents = []
                selectedParents.append(rankedPopulation[rankSelected1][0][:])
                selectedParents.append(rankedPopulation[rankSelected2][0][:])
                crossoverTest = random.randint(0, 10)

                ## Crossover operations
                if crossoverTest <=7:
                    child1 = []
                    child2 = []
                    for j in range(0,len(selectedParents[0])):
                        randomTest = random.randint(0, 1)
                        if randomTest is 0:
                            child1.append(selectedParents[0][j])
                            child2.append(selectedParents[1][j])
                        elif randomTest is 1:
                            child1.append(selectedParents[1][j])
                            child2.append(selectedParents[0][j])
                    newPopulation.append(child1[:])
                    newPopulation.append(child2[:])
                else:
                    newPopulation.append(selectedParents[0][:])
                    newPopulation.append(selectedParents[1][:])

            ##Mutation operations
            for i in range(0,len(newPopulation)):
                mutationTest = random.randint(0, 10)
                if mutationTest <= 1:
                    selectedAction = random.randint(0, len(newPopulation[i])-1)
                    newPopulation[i][selectedAction] = possible[random.randint(0, len(possible) - 1)]

            population = newPopulation
        return bestScore[0]
    ##Rank selection function
    def rankSelection(self, randomNumber):
        if 1 <= randomNumber <= 8:
            return 7
        elif 9 <= randomNumber <= 15:
            return 6
        elif 16 <= randomNumber <= 21:
            return 5
        elif 22 <= randomNumber <= 26:
            return 4
        elif 27 <= randomNumber <= 30:
            return 3
        elif 31 <= randomNumber <= 33:
            return 2
        elif 34 <= randomNumber <= 35:
            return 1
        elif randomNumber is 36:
            return 0



class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
            global noneReached
            noneReached = False
            thisNode = Node(None)                   ##Generate Root Node
            while noneReached is False:
                searchedNode = self.MCTSTreeFormation(thisNode, state)
                if searchedNode is None:
                    break
                elif searchedNode == 0:             ## Already back propogated on Terminal State
                    continue
                score = self.defaultRollout(searchedNode, state)
                if score is None:
                    break
                elif score == 0:                    ## Already back propogated on Terminal State
                    continue
                self.backProp(searchedNode, score)
            mostVisits = max([node.visitNumber for node in thisNode.children])  # get children with most visits
            bestActions = [nodes.action for nodes in thisNode.children if nodes.visitNumber == mostVisits]
            # return random action from the list of the best actions
            return random.choice(bestActions)


    ##MCTS Back Propogation
    def backProp(self, node, score):
        while node is not None:
            node.visitNumber = node.visitNumber + 1
            node.score = node.score + score
            node = node.parent
        return

    ##MCTS Default Policy
    def defaultRollout(self, node, rootState):
        nodeSet = []
        tempNode = node
        ##Backup from current node to root
        while tempNode.parent is not None:
            nodeSet.append(tempNode)
            tempNode = tempNode.parent
        nodeSet = nodeSet[::-1]
        tempState = rootState
        ##Move to current state from root State
        for i in nodeSet:
            prevState = tempState
            tempState = tempState.generatePacmanSuccessor(i.action)
            if tempState is None:
                self.backProp(i, normalizedScoreEvaluation(rootState, prevState))
                return None
            if tempState.isWin() + tempState.isLose() != 0:
                self.backProp(i, normalizedScoreEvaluation(rootState, tempState))
                return 0
        newState = tempState

        ##Rollout upto 5 states
        for j in range(0,5):
            if newState.isWin() + newState.isLose() == 0:
                legal = newState.getLegalPacmanActions()
                if not legal:
                    break
                prevState = newState
                newState = newState.generatePacmanSuccessor(random.choice(legal))
                if newState is None:
                    newState = prevState
                    global noneReached
                    noneReached = True
                    break
            else:
                break
        qScore = normalizedScoreEvaluation(rootState, newState)
        return qScore

    ##MCTS Selection Function
    def selectionFormuala(self, node):
        bestScore = -20000
        topChildren = []                        ##Children with best UCT score
        for i in node.children:
            result = (i.score/i.visitNumber) + 1*math.sqrt((2*math.log(node.visitNumber))/i.visitNumber)
            if result == bestScore:
                topChildren.append(i)
            if result > bestScore:
                bestScore = result
                topChildren = [i]
        if topChildren:                         ##Choose random child with same UCT score
            return topChildren[random.randint(0, len(topChildren) - 1)]
        else:
            return node.children[random.randint(0, len(node.children) - 1)]

    ##MCTS Expand Function
    def expansion(self, node, rootState):
        nodeSet = []
        tempNode = node
        ##Backup from current node to root
        while tempNode.parent is not None:
            nodeSet.append(tempNode)
            tempNode = tempNode.parent

        nodeSet = nodeSet[::-1]
        tempState = rootState

        ##Move to current state from root state
        for i in nodeSet:
            prevState = tempState
            tempState = tempState.generatePacmanSuccessor(i.action)
            if tempState is None:
                self.backProp(i, normalizedScoreEvaluation(rootState, prevState))
                return None
            if tempState.isWin() + tempState.isLose() != 0:
                self.backProp(i, normalizedScoreEvaluation(rootState, tempState))
                return 0

        expandedChildrenActions = [i.action for i in node.children]     ##Get already expanded actions
        legal = tempState.getLegalPacmanActions()
        for j in legal:
            if j not in expandedChildrenActions:            ##Add only those children that have not yet been expanded
                node.addNewChild(j)
                break
        if len(node.children) == len(legal):                ##if children have all legal actions then its fully expanded
            node.allStatesExpanded = True
        return node.children[-1]

    ##MCTS Tree Policy
    def MCTSTreeFormation(self, node, rootState):
        foundUnexpandedChild = False
        while foundUnexpandedChild is False:
            if node.allStatesExpanded is True:
                node = self.selectionFormuala(node)         ##Select and replace
            else:
                foundUnexpandedChild = True                 ##Expand and return
        return self.expansion(node, rootState)

##MCTS Class for each node in Tree
class Node():

    def __init__(self, action, parent=None):
        self.children = []
        self.visitNumber = 1
        self.allStatesExpanded = False
        self.action = action
        self.parent = parent
        self.score = 0

    def addNewChild(self, action):
        newChild = Node(action, self)
        self.children.append(newChild)

    def expansionDone(self):
        if self.allStatesExpanded is True:
            return True
        else:
            return False

class RevampedAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write your algorithm Algorithm instead of returning Directions.STOP
        global lastAction
        global capsules
        global ghostPositions
        global chaseStepCount
        global capsuleHit
        global ghostEaten
        ghostPositions = state.getGhostPositions()[:]
        currX, currY = state.getPacmanPosition()
        pellets = state.getPellets()
        currentScore = scoreEvaluation(state)
        countReached = False  # Flag to check if call limit is reached
        outputSuccessors = []  # Final successor list with fringe nodes
        winNodes = []
        direction = Directions.STOP
        depth = 1
        actionDict = {'West': 0, 'North': 0, 'East': 0, 'South': 0}
        legal = state.getLegalPacmanActions()
        successors = [(state.generatePacmanSuccessor(action), action, depth) for action in legal]

        ##Chase only ghosts logic
        if capsules is not None and (currX, currY) in capsules:
            capsuleHit = True
        if capsuleHit == True:
            chaseGhosts = []
            closestGhost = findClosestPoint((currX, currY), ghostPositions)
            for i in successors:
                if i[0].getScore() < state.getScore():           ##If score after eating is lower, break and continue A*
                    break
                if ghostEaten == len(ghostPositions):            ##If all ghosts are eaten stop chasing
                    chaseStepCount = 30
                    capsuleHit = False
                    ghostEaten = 0
                    break
                if i[0].getScore() >= (state.getScore() + 200):
                    ghostEaten += 1
                    return i[1]
                dist = getManhattanDistance(i[0].getPacmanPosition(), closestGhost)    ##get manhattan distance to ghost
                chaseGhosts.append((dist, i[1]))
            chaseGhosts = sorted(chaseGhosts, key=lambda distance: distance[0])        ##get minimum Manhattan distance
            chaseStepCount -= 1
            if chaseStepCount <= 0:                     ##Stop chasing after 30 steps
                capsuleHit = False
                chaseStepCount = 30
                ghostEaten = 0
            if chaseGhosts and chaseGhosts != []:
                return chaseGhosts[0][1]

        capsules = state.getCapsules()

        ## Check for single pellet from pacman neighbour positions
        pelletCoordinate = checkAllNeighbours((currX, currY), pellets, state)
        if pelletCoordinate != None and pelletCoordinate not in ghostPositions:
            actionDict = {'West': 0, 'North': 0, 'East': 0, 'South': 0}
            nbrX, nbrY = pelletCoordinate
            if nbrX == currX - 1:                                    ##Add weightage to action leading to single pellet
                # return 'West'
                actionDict['West'] = 20
            elif nbrX == currX + 1:
                # return 'East'
                actionDict['East'] = 20
            elif nbrY == currY - 1:
                # return 'South'
                actionDict['South'] = 20
            elif nbrY == currY + 1:
                # return 'North'
                actionDict['North'] = 20

        while (successors):  # A* Loop
            for i in range(len(successors)):
                if successors[i][0] != None:
                    childNode, childDirection, childDepth = successors[i]
                    scoreEval = scoreEvaluation(childNode)
                    if childNode.isWin() == True:
                        winNodes.append((scoreEval, childDirection))           ##Save all states leading to win state
                    elif childNode.isLose() == True:
                        continue  # if lose state, do not add to outputSuccessor list
                    # Add remaining states to outputSuccessor list with their path cost
                    pathCost = childDepth - (scoreEval + actionDict.get(childDirection) - scoreEvaluation(state))
                    outputSuccessors.append((childNode, childDirection, childDepth, pathCost, scoreEval))
                else:
                    countReached = True

            # Sort outputSuccessor list to find successor states of node with minimum path cost and pop it from list
            if countReached == False:
                outputSuccessors = sorted(outputSuccessors, key=lambda cost: cost[3])
                if not outputSuccessors or outputSuccessors == None:
                    return direction
                minState, direction, minNodeDepth, minNodeCost, minNodeScore = outputSuccessors[0]
                legal = minState.getLegalPacmanActions()
                depth = minNodeDepth + 1
                if depth >=25:                          ##Limit tree depth to 25
                    break
                outputSuccessors.pop(0)
                successors = [(minState.generatePacmanSuccessor(action), direction, depth) for action in legal]
            else:
                successors = []
        # Return best scoring action leading to win state, if any
        if winNodes or winNodes != []:
            bestScore = max(winNodes)[0]
            # get all actions that lead to the highest score
            bestActions = [pair[1] for pair in winNodes if pair[0] == bestScore]
            # return random action from the list of the best actions
            return random.choice(bestActions)
        # Return action leading to shallowest fringe node with best score
        if not outputSuccessors or outputSuccessors == None:
            return direction
        bestScore = max(outputSuccessors)[4]
        ## When no pellets are near till end of forward modal best score will be same as current state score
        ## Take manhattan distance as heurestic in this case to prevent pacman from getting stuck
        if bestScore == currentScore:
            legal = state.getLegalPacmanActions()
            scoredList = []
            for action in legal:
                if action == 'West':
                    newX = currX - 1
                    newY = currY
                elif action == 'East':
                    newX = currX + 1
                    newY = currY
                elif action == 'North':
                    newX = currX
                    newY = currY + 1
                elif action == 'South':
                    newX = currX
                    newY = currY - 1
                if (newX, newY) not in ghostPositions and (newX + 1, newY) not in ghostPositions \
                        and (newX, newY - 1) not in ghostPositions and (newX, newY + 1) not in ghostPositions \
                        and (newX - 1, newY) not in ghostPositions:
                    mhtnDistance = manhattanDistance((newX, newY), pellets)
                    if (newX, newY) == lastAction:
                        mhtnDistance = mhtnDistance + 60
                    scoredList.append((mhtnDistance, action))
            minDist = min(scoredList)[0]
            finalActions = [pair[1] for pair in scoredList if pair[0] == minDist]
            lastAction = (currX, currY)
            return finalActions[random.randint(0, len(finalActions) - 1)]
        bestActions = [(pair[1], pair[2]) for pair in outputSuccessors if pair[4] == bestScore]
        leastDepth = min(bestActions)[1]
        finalActions = [pair[0] for pair in bestActions if pair[1] == leastDepth]
        return finalActions[0]

##Get sum of manhattan distances of all pellets from Pacman
def manhattanDistance((x1,y1), pelletList):
    totalDistance = 0
    for x2,y2 in pelletList:
        totalDistance += abs(x1-x2) + abs(y1-y2)
    return totalDistance

##Check all neighbours of Pacman for a single isolated pellet and return its position if any
def checkAllNeighbours((x1,y1), list, state):
    for i in getAllNeighbourPositions((x1,y1)):
        if i in list:
            if i in state.getGhostPositions():
                continue
            x2,y2 = i
            nbhrSet = getAllNeighbourPositions((x2, y2))
            if nbhrSet[0] not in list and nbhrSet[1] not in list and nbhrSet[2] not in list and nbhrSet[3] not in list:
                return (x2,y2)
    return None

def getAllNeighbourPositions((x1,y1)):
    return [(x1-1,y1), (x1+1,y1), (x1,y1-1), (x1, y1+1)]

##Get closest chost position from pacman
def findClosestPoint(point, ghosts):
    minDistance = (9999999,None)
    x1,y1 = point
    for x2, y2 in ghosts:
        totalDistance = abs(x1 - x2) + abs(y1 - y2)
        if totalDistance < minDistance[0]:
            minDistance = (totalDistance,(x2,y2))
    return minDistance[1]

def getManhattanDistance(point1, point2):
    x1,y1 = point1
    x2,y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)

