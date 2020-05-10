
from skXCS.DataManagement import DataManagement

class Environment:
    def __init__(self,X,y,xcs):
        self.dataRef = 0
        self.formatData = DataManagement(X,y,xcs)
        self.max_payoff = xcs.max_payoff

        self.currentTrainState = self.formatData.trainFormatted[0][self.dataRef]
        self.currentTrainPhenotype = self.formatData.trainFormatted[1][self.dataRef]

    def getTrainState(self):
        return self.currentTrainState

    def newInstance(self):
        if self.dataRef < self.formatData.numTrainInstances-1:
            self.dataRef+=1
            self.currentTrainState = self.formatData.trainFormatted[0][self.dataRef]
            self.currentTrainPhenotype = self.formatData.trainFormatted[1][self.dataRef]
        else:
            self.resetDataRef()

    def resetDataRef(self):
        self.dataRef = 0
        self.currentTrainState = self.formatData.trainFormatted[0][self.dataRef]
        self.currentTrainPhenotype = self.formatData.trainFormatted[1][self.dataRef]

    def executeAction(self,action):
        if action == self.currentTrainPhenotype:
            return self.max_payoff
        return 0

