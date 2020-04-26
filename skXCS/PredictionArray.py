import random
import numpy as np

class PredictionArray:
    def __init__(self,population,xcs):
        self.predictionArray = {}
        self.fitnesses = {}
        self.actionList = xcs.env.formatData.phenotypeList
        self.probabilities = {}
        self.hasMatch = len(population.matchSet) != 0

        for eachClass in self.actionList:
            self.predictionArray[eachClass] = 0.0
            self.fitnesses[eachClass] = 0.0

        for ref in population.matchSet:
            cl = population.popSet[ref]
            self.predictionArray[cl.action] += cl.prediction*cl.fitness
            self.fitnesses[cl.action] += cl.fitness

        for eachClass in self.actionList:
            if self.fitnesses[eachClass] != 0:
                self.predictionArray[eachClass] /= self.fitnesses[eachClass]
            else:
                self.predictionArray[eachClass] = 0

        #Populate Probabilities
        probabilitySum = 0
        for action,value in sorted(self.predictionArray.items()):
            self.probabilities[action] = value
            probabilitySum += value
        if probabilitySum == 0:
            for action, prob in sorted(self.probabilities.items()):
                self.probabilities[action] = 0
        else:
            for action, prob in sorted(self.probabilities.items()):
                self.probabilities[action] = prob/probabilitySum

    def getBestValue(self):
        return max(self.predictionArray,key=self.predictionArray.get)

    def getValue(self,action):
        return self.predictionArray[action]

    ##*************** Action selection functions ****************
    def randomActionWinner(self):
        """ Selects an action randomly. The function assures that the chosen action is represented by at least one classifier. """
        while True:
            ret = random.choice(self.actionList)
            if self.fitnesses[ret] != 0:
                break
        return ret

    def bestActionWinner(self):
        """ Selects the action in the prediction array with the best value.
         *MODIFIED so that in the case of a tie between actions - an action is selected randomly between the tied highest actions. """
        highVal = 0.0
        for action,value in self.predictionArray.items():
            if value > highVal:
                highVal = value
        bestIndexList = []
        for action,value in self.predictionArray.items():
            if value == highVal:
                bestIndexList.append(action)
        return random.choice(bestIndexList)

    ##*************** Get ActionProbabilities ****************
    def getProbabilities(self):
        probabilityList = np.empty(len(sorted(self.probabilities.items())))
        counter = 0
        for action,prob in sorted(self.probabilities.items()):
            probabilityList[counter] = prob
            counter += 1
        return probabilityList