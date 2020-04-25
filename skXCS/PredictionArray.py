import random
class PredictionArray:
    def __init__(self,population,xcs):
        self.predictionArray = {}
        self.fitnesses = {}
        self.actionList = xcs.env.formatData.phenotypeList

        for eachClass in self.actionList:
            self.predictionArray[eachClass] = 0.0
            self.fitnesses[eachClass] = 0.0
            self.tieBreak_Numerosity[eachClass] = 0.0
            self.tieBreak_TimeStamp[eachClass] = 0.0

        for ref in population.matchSet:
            cl = population.popSet[ref]
            self.predictionArray[cl.action] += cl.prediction*cl.fitness
            self.fitnesses[cl.action] += cl.fitness

        for eachClass in self.actionList:
            if self.fitnesses[eachClass] != 0:
                self.predictionArray[eachClass] /= self.fitnesses[eachClass]
            else:
                self.predictionArray[eachClass] = 0

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

