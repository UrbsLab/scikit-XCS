import random
import copy

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
        self.matchCount = 0

        self.actionSetSize = None
        self.timestampGA = xcs.iterationCount
        self.initTimeStamp = xcs.iterationCount
        self.deletionProb = None

        pass

    def initializeWithParentClassifier(self,classifier):
        self.specifiedAttList = copy.deepcopy(classifier.specifiedAttList)
        self.condition = copy.deepcopy(classifier.condition)
        self.action = copy.deepcopy(classifier.action)

        self.actionSetSize = classifier.actionSetSize
        self.prediction = classifier.prediction
        self.predictionError = classifier.predictionError
        self.fitness = classifier.fitness/classifier.numerosity

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
            attributeInfoValue = xcs.env.formatData.attributeInfoContinuous[attRef]

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
        if classifier.action == self.action and len(classifier.specifiedAttList) == len(self.specifiedAttList):
            clRefs = sorted(classifier.specifiedAttList)
            selfRefs = sorted(self.specifiedAttList)
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
            self.predictionError = self.predictionError + (abs(P - self.prediction) - self.predictionError) / float(self.experience)
        else:
            self.predictionError = self.predictionError + xcs.beta * (abs(P - self.prediction) - self.predictionError)

    def updatePrediction(self,P,xcs):
        if self.experience < 1.0 / xcs.beta:
            self.prediction = self.prediction + (P-self.prediction) / float(self.experience)
        else:
            self.prediction = self.prediction + xcs.beta * (P - self.prediction)

    def updateActionSetSize(self,numerositySum,xcs):
        if self.experience < 1.0/xcs.beta:
            self.actionSetSize = self.actionSetSize + (numerositySum - self.actionSetSize) / float(self.experience)
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

    def subsumes(self,classifier,xcs):
        return self.action == classifier.action and self.isSubsumer(xcs) and self.isMoreGeneral(classifier,xcs)

    def updateTimestamp(self,timestamp):
        self.timestampGA = timestamp

    def uniformCrossover(self,classifier,xcs):
        p_self_specifiedAttList = copy.deepcopy(self.specifiedAttList)
        p_cl_specifiedAttList = copy.deepcopy(classifier.specifiedAttList)

        # Make list of attribute references appearing in at least one of the parents.-----------------------------
        comboAttList = []
        for i in p_self_specifiedAttList:
            comboAttList.append(i)
        for i in p_cl_specifiedAttList:
            if i not in comboAttList:
                comboAttList.append(i)
            elif not xcs.env.formatData.attributeInfoType[i]:
                comboAttList.remove(i)
        comboAttList.sort()

        changed = False
        for attRef in comboAttList:
            attributeInfoType = xcs.env.formatData.attributeInfoType[attRef]
            probability = 0.5
            ref = 0
            if attRef in p_self_specifiedAttList:
                ref += 1
            if attRef in p_cl_specifiedAttList:
                ref += 1

            if ref == 0:
                pass
            elif ref == 1:
                if attRef in p_self_specifiedAttList and random.random() > probability:
                    i = self.specifiedAttList.index(attRef)
                    classifier.condition.append(self.condition.pop(i))

                    classifier.specifiedAttList.append(attRef)
                    self.specifiedAttList.remove(attRef)
                    changed = True

                if attRef in p_cl_specifiedAttList and random.random() < probability:
                    i = classifier.specifiedAttList.index(attRef)
                    self.condition.append(classifier.condition.pop(i))

                    self.specifiedAttList.append(attRef)
                    classifier.specifiedAttList.remove(attRef)
                    changed = True
            else:
                # Continuous Attribute
                if attributeInfoType:
                    i_cl1 = self.specifiedAttList.index(attRef)
                    i_cl2 = classifier.specifiedAttList.index(attRef)
                    tempKey = random.randint(0, 3)
                    if tempKey == 0:
                        temp = self.condition[i_cl1][0]
                        self.condition[i_cl1][0] = classifier.condition[i_cl2][0]
                        classifier.condition[i_cl2][0] = temp
                    elif tempKey == 1:
                        temp = self.condition[i_cl1][1]
                        self.condition[i_cl1][1] = classifier.condition[i_cl2][1]
                        classifier.condition[i_cl2][1] = temp
                    else:
                        allList = self.condition[i_cl1] + classifier.condition[i_cl2]
                        newMin = min(allList)
                        newMax = max(allList)
                        if tempKey == 2:
                            self.condition[i_cl1] = [newMin, newMax]
                            classifier.condition.pop(i_cl2)

                            classifier.specifiedAttList.remove(attRef)
                        else:
                            classifier.condition[i_cl2] = [newMin, newMax]
                            self.condition.pop(i_cl1)

                            self.specifiedAttList.remove(attRef)

                # Discrete Attribute
                else:
                    pass

        tempList1 = copy.deepcopy(p_self_specifiedAttList)
        tempList2 = copy.deepcopy(classifier.specifiedAttList)
        tempList1.sort()
        tempList2.sort()

        if changed and len(set(tempList1) & set(tempList2)) == len(tempList2):
            changed = False

        return changed

    def mutation(self,state,xcs):
        changedByConditionMutation = self.mutateCondition(state,xcs)
        changedByActionMutation = self.mutateAction(xcs)
        return changedByConditionMutation or changedByActionMutation

    def mutateCondition(self,state,xcs):
        changed = False
        for attRef in range(xcs.env.formatData.numAttributes):
            attributeInfoType = xcs.env.formatData.attributeInfoType[attRef]
            if attributeInfoType:
                attributeInfoValue = xcs.env.formatData.attributeInfoContinuous[attRef]

            if random.random() < xcs.p_mutation and not(state[attRef] == None):
                if not (attRef in self.specifiedAttList):
                    self.specifiedAttList.append(attRef)
                    self.createMatchingAttribute(xcs,attRef,state)
                    changed = True
                elif attRef in self.specifiedAttList:
                    i = self.specifiedAttList.index(attRef)

                    if not attributeInfoType or random.random() > 0.5:
                        del self.specifiedAttList[i]
                        del self.condition[i]
                        changed = True
                    else:
                        attRange = float(attributeInfoValue[1]) - float(attributeInfoValue[0])
                        mutateRange = random.random() * 0.5 * attRange
                        if random.random() > 0.5:
                            if random.random() > 0.5:
                                self.condition[i][0] += mutateRange
                            else:
                                self.condition[i][0] -= mutateRange
                        else:
                            if random.random() > 0.5:
                                self.condition[i][1] += mutateRange
                            else:
                                self.condition[i][1] -= mutateRange
                        self.condition[i] = sorted(self.condition[i])
                        changed = True
                else:
                    pass
        return changed

    def mutateAction(self,xcs):
        changed = False
        if random.random() < xcs.p_mutation:
            action = random.choice(xcs.env.formatData.phenotypeList)
            while action == self.action:
                action = random.choice(xcs.env.formatData.phenotypeList)
            self.action = action
            changed = True
        return changed

    def getDelProp(self,meanFitness,xcs):
        if self.fitness / self.numerosity >= xcs.delta * meanFitness or self.experience < xcs.theta_del:
            deletionVote = self.actionSetSize * self.numerosity

        elif self.fitness == 0.0:
            deletionVote = self.actionSetSize * self.numerosity * meanFitness / (xcs.init_fit / self.numerosity)
        else:
            deletionVote = self.actionSetSize * self.numerosity * meanFitness / (self.fitness / self.numerosity)
        return deletionVote