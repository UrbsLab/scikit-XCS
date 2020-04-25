import random

class Classifier:
    def __init__(self,xcs):
        self.specifiedAttList = []
        self.condition = []
        self.action = None

        self.prediction = xcs.init_prediction
        self.fitness = xcs.init_fitness
        self.predictionError = xcs.init_e

        self.numerosity = 1
        self.experience = 0 #aka action set count
        self.actionSetSize = None
        self.timestampGA = xcs.iterationCount
        self.initTimeStamp = xcs.iterationCount


        pass

    def match(self,state,xcs):
        for i in range(len(self.condition)):
            specifiedIndex = self.specifiedAttList[i]
            attributeInfoType = xcs.env.formatData.attributeInfoType[specifiedIndex]
            instanceValue = state[specifiedIndex]

            #Continuous
            if attributeInfoType:
                if instanceValue == None:
                    return False
                elif self.condition[i][0] < instanceValue < self.condition[i][1]:
                    pass
                else:
                    return False
            else:
                if instanceValue == self.condition[i]:
                    pass
                elif instanceValue == None:
                    return False
                else:
                    return False
        return True

    def initializeWithMatchingStateAndGivenAction(self,setSize,state,action,xcs):
        self.action = action
        self.actionSetSize = setSize

        while len(self.specifiedAttList) < 1:
            for attRef in range(len(state)):
                if random.random() > xcs.p_general and not(state[attRef] == None):
                    self.specifiedAttList.append(attRef)
                    self.createMatchingAttribute(xcs,attRef,state)


    def createMatchingAttribute(self,xcs,attRef,state):
        attributeInfoType = xcs.env.formatData.attributeInfoType[attRef]
        if attributeInfoType:
            attributeInfoValue = elcs.env.formatData.attributeInfoContinuous[attRef]

        # Continuous attribute
        if attributeInfoType:
            attRange = attributeInfoValue[1] - attributeInfoValue[0]
            rangeRadius = random.randint(25, 75) * 0.01 * attRange / 2.0  # Continuous initialization domain radius.
            ar = state[attRef]
            Low = ar - rangeRadius
            High = ar + rangeRadius
            condList = [Low, High]
            self.condition.append(condList)

        # Discrete attribute
        else:
            condList = state[attRef]
            self.condition.append(condList)

    def equals(self,classifier):
        if classifier.action == self.action and len(classifier.specifiedAttList) == len(classifier.specifiedAttList):
            clRefs = sorted(classifier.specifiedAttList)
            selfRefs = sorted(classifier.specifiedAttList)
            if clRefs == selfRefs:
                for i in range(len(classifier.specifiedAttList)):
                    tempIndex = self.specifiedAttList.index(classifier.specifiedAttList[i])
                    if not (classifier.condition[i] == self.condition[tempIndex]):
                        return False
                return True
        return False

    def updateNumerosity(self,num):
        self.numerosity += num

    def increaseExperience(self):
        self.experience += 1

    def updatePredictionError(self,P,xcs):
        if self.experience < 1.0/xcs.beta:
            self.predictionError = (self.predictionError*(self.experience - 1) + abs(P - self.prediction)) / float(self.experience)
        else:
            self.predictionError = self.predictionError + xcs.beta * (abs(P - self.prediction) - self.predictionError)

    def updatePrediction(self,P,xcs):
        if self.experience < 1.0 / xcs.beta:
            self.prediction = (self.prediction * (self.experience - 1) + P) / float(self.experience)
        else:
            self.prediction = self.prediction + xcs.beta * (P - self.prediction)

    def updateActionSetSize(self,numerositySum,xcs):
        if self.experience < 1.0/xcs.beta:
            self.actionSetSize = (self.actionSetSize * (self.experience-1)+ numerositySum) / float(self.experience)
        else:
            self.actionSetSize = self.actionSetSize + xcs.beta * (numerositySum - self.actionSetSize)

    def getAccuracy(self,xcs):
        """ Returns the accuracy of the classifier.
        The accuracy is determined from the prediction error of the classifier using Wilson's
        power function as published in 'Get Real! XCS with continuous-valued inputs' (1999) """

        if self.predictionError <= xcs.e_0:
            accuracy = 1.0
        else:
            accuracy = xcs.alpha * ((self.predictionError / xcs.e_0) ** (-xcs.nu))

        return accuracy

    def updateFitness(self, accSum, accuracy,xcs):
        """ Updates the fitness of the classifier according to the relative accuracy.
        @param accSum The sum of all the accuracies in the action set
        @param accuracy The accuracy of the classifier. """

        self.fitness = self.fitness + xcs.beta * ((accuracy * self.numerosity) / float(accSum) - self.fitness)

    def isSubsumer(self,xcs):
        """ Returns if the classifier is a possible subsumer. It is affirmed if the classifier
                has a sufficient experience and if its reward prediction error is sufficiently low.  """

        if self.experience > xcs.theta_sub and self.predictionError < xcs.e_0:
            return True
        return False

    def isMoreGeneral(self,classifier,xcs):
        if len(self.specifiedAttList) >= len(classifier.specifiedAttList):
            return False
        for i in range(len(self.specifiedAttList)):
            if self.specifiedAttList[i] not in classifier.specifiedAttList:
                return False

            attributeInfoType = xcs.env.formatData.attributeInfoType[self.specifiedAttList[i]]
            if attributeInfoType:
                otherRef = classifier.specifiedAttList.index(self.specifiedAttList[i])
                if self.condition[i][0] < classifier.condition[otherRef][0]:
                    return False
                if self.condition[i][1] > classifier.condition[otherRef][1]:
                    return False
        return True