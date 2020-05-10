
import copy
import random
from skXCS.Classifier import Classifier

class ClassifierSet:
    def __init__(self):
        self.popSet = []
        self.matchSet = []
        self.actionSet = []
        self.microPopSize = 0

    ####Match Set Creation####
    def createMatchSet(self,state,xcs):
        xcs.timer.startTimeMatching()
        actionsNotCovered = copy.deepcopy(xcs.env.formatData.phenotypeList)
        totalNumActions = len(xcs.env.formatData.phenotypeList)

        matchSetSize = 0
        for i in range(len(self.popSet)):
            classifier = self.popSet[i]
            if classifier.match(state,xcs):
                self.matchSet.append(i)
                matchSetSize+=classifier.numerosity
                if classifier.action in actionsNotCovered:
                    actionsNotCovered.remove(classifier.action)

        if xcs.env.formatData.isBinaryClassification:
            doCovering = totalNumActions - len(actionsNotCovered) < xcs.theta_matching or len(self.matchSet) < 5 #Second condition only holds for 1 covering round
        else:
            doCovering = totalNumActions - len(actionsNotCovered) < xcs.theta_matching

        while doCovering:
            if len(actionsNotCovered) != 0:
                action = random.choice(actionsNotCovered)
            else:
                action = random.choice(copy.deepcopy(xcs.env.formatData.phenotypeList))
            coveredClassifier = Classifier(xcs)
            coveredClassifier.initializeWithMatchingStateAndGivenAction(matchSetSize,state,action,xcs)
            self.addClassifierToPopulation(xcs,coveredClassifier,True)
            self.matchSet.append(len(self.popSet)-1)
            if len(actionsNotCovered) != 0:
                actionsNotCovered.remove(action)
            xcs.trackingObj.coveringCount += 1

            doCovering = totalNumActions - len(actionsNotCovered) < xcs.theta_matching

        for ref in self.matchSet:
            self.popSet[ref].matchCount += 1
        xcs.timer.stopTimeMatching()

    def getIdenticalClassifier(self,xcs,newClassifier):
        for classifier in self.popSet:
            if newClassifier.equals(classifier):
                return classifier
        return None

    def addClassifierToPopulation(self,xcs,classifier,isCovering):
        oldCl = None
        if not isCovering:
            oldCl = self.getIdenticalClassifier(xcs,classifier)
        if oldCl != None:
            oldCl.updateNumerosity(1)
            self.microPopSize += 1
        else:
            self.popSet.append(classifier)
            self.microPopSize += 1

    ####Action Set Creation####
    def createActionSet(self,action):
        for ref in self.matchSet:
            if self.popSet[ref].action == action:
                self.actionSet.append(ref)

    ####Update Action Set Statistics####
    def updateActionSet(self,reward,xcs):
        P = reward

        actionSetNumerositySum = 0
        for i in self.actionSet:
            ref = self.popSet[i]
            actionSetNumerositySum += ref.numerosity

        for cl in self.actionSet:
            classifier = self.popSet[cl]
            classifier.increaseExperience()
            classifier.updatePrediction(P,xcs)
            classifier.updatePredictionError(P,xcs)
            classifier.updateActionSetSize(actionSetNumerositySum,xcs)

        self.updateFitnessSet(xcs)
        if xcs.do_action_set_subsumption:
            xcs.timer.startTimeSubsumption()
            self.do_action_set_subsumption(xcs)
            xcs.timer.stopTimeSubsumption()

    def updateFitnessSet(self,xcs):
        accuracySum = 0
        accuracies = []

        i = 0
        for clRef in self.actionSet:
            classifier = self.popSet[clRef]
            accuracies.append(classifier.getAccuracy(xcs))
            accuracySum = accuracySum + accuracies[i]*classifier.numerosity
            i+=1

        i = 0
        for clRef in self.actionSet:
            classifier = self.popSet[clRef]
            classifier.updateFitness(accuracySum,accuracies[i],xcs)
            i+=1

    ####Action Set Subsumption####
    def do_action_set_subsumption(self,xcs):
        subsumer = None
        for clRef in self.actionSet:
            classifier = self.popSet[clRef]
            if classifier.isSubsumer(xcs):
                if subsumer == None or classifier.isMoreGeneral(subsumer,xcs):
                    subsumer = classifier

        if subsumer != None:
            i = 0
            while i < len(self.actionSet):
                ref = self.actionSet[i]
                if subsumer.isMoreGeneral(self.popSet[ref],xcs):
                    xcs.trackingObj.subsumptionCount += 1
                    subsumer.updateNumerosity(self.popSet[ref].numerosity)
                    self.removeMacroClassifier(ref)
                    self.deleteFromMatchSet(ref)
                    self.deleteFromActionSet(ref)
                    i -= 1
                i+=1

    def removeMacroClassifier(self, ref):
        del self.popSet[ref]

    def deleteFromMatchSet(self, deleteRef):
        if deleteRef in self.matchSet:
            self.matchSet.remove(deleteRef)

        for j in range(len(self.matchSet)):
            ref = self.matchSet[j]
            if ref > deleteRef:
                self.matchSet[j] -= 1

    def deleteFromActionSet(self, deleteRef):
        if deleteRef in self.actionSet:
            self.actionSet.remove(deleteRef)

        for j in range(len(self.actionSet)):
            ref = self.actionSet[j]
            if ref > deleteRef:
                self.actionSet[j] -= 1

    ####GA####
    def runGA(self,state,xcs):
        #GA Run Requirement
        if (xcs.iterationCount - self.getIterStampAverage()) < xcs.theta_GA:
            return

        xcs.timer.startTimeGA()
        self.setIterStamps(xcs.iterationCount)
        parentClassifiers = self.selectTwoParentViaTournament(xcs)
        parentClassifier1 = parentClassifiers[0]
        parentClassifier2 = parentClassifiers[1]

        childClassifier1 = Classifier(xcs)
        childClassifier1.initializeWithParentClassifier(parentClassifier1)
        childClassifier2 = Classifier(xcs)
        childClassifier2.initializeWithParentClassifier(parentClassifier2)

        changedByCrossover = False
        if not childClassifier1.equals(childClassifier2) and random.random() < xcs.p_crossover:
            changedByCrossover = childClassifier1.uniformCrossover(childClassifier2,xcs)

        if changedByCrossover:
            childClassifier1.prediction = (childClassifier1.prediction + childClassifier2.prediction)/2
            childClassifier2.predictionError = xcs.prediction_error_reduction*(childClassifier1.predictionError + childClassifier2.predictionError)/2
            childClassifier1.fitness = xcs.fitness_reduction*(childClassifier1.fitness+childClassifier2.fitness)/2
            childClassifier2.prediction = childClassifier1.prediction
            childClassifier2.predictionError = childClassifier1.predictionError
            childClassifier2.fitness = childClassifier1.fitness
        else:
            childClassifier1.fitness = xcs.fitness_reduction * childClassifier1.fitness
            childClassifier2.fitness = xcs.fitness_reduction * childClassifier2.fitness

        changedByMutation1 = childClassifier1.mutation(state,xcs)
        changedByMutation2 = childClassifier2.mutation(state,xcs)
        xcs.timer.stopTimeGA()

        if changedByMutation1 or changedByMutation2 or changedByCrossover:
            if changedByMutation1 or changedByMutation2:
                xcs.trackingObj.mutationCount += 1
            if changedByCrossover:
                xcs.trackingObj.crossOverCount += 1
            self.insertDiscoveredClassifiers(childClassifier1,childClassifier2,parentClassifier1,parentClassifier2,xcs)

    def insertDiscoveredClassifiers(self,child1,child2,parent1,parent2,xcs):
        if xcs.do_GA_subsumption:
            xcs.timer.startTimeSubsumption()
            self.subsumeClassifier(child1,parent1,parent2,xcs)
            self.subsumeClassifier(child2,parent1,parent2,xcs)
            xcs.timer.stopTimeSubsumption()
        else:
            if len(child1.specifiedAttList) > 0:
                self.addClassifierToPopulation(xcs, child1, False)
            if len(child2.specifiedAttList) > 0:
                self.addClassifierToPopulation(xcs, child2, False)

    def subsumeClassifier(self,child,parent1,parent2,xcs):
        if parent1.subsumes(child,xcs):
            self.microPopSize += 1
            parent1.updateNumerosity(1)
            xcs.trackingObj.subsumptionCount += 1
        elif parent2.subsumes(child,xcs):
            self.microPopSize += 1
            parent2.updateNumerosity(1)
            xcs.trackingObj.subsumptionCount += 1
        else: #No additional [A] subsumption w/ offspring rules
            if len(child.specifiedAttList) > 0:
                self.addClassifierToPopulation(xcs, child, False)

    def getIterStampAverage(self): #Average GA Timestamp
        sumCl = 0
        numSum = 0
        for ref in self.actionSet:
            sumCl += self.popSet[ref].timestampGA * self.popSet[ref].numerosity
            numSum += self.popSet[ref].numerosity
        if numSum != 0:
            return sumCl/float(numSum)
        else:
            return 0

    def getInitStampAverage(self): #Average Init Timestamp
        sumCl = 0
        numSum = 0
        for ref in self.actionSet:
            sumCl += self.popSet[ref].initTimeStamp * self.popSet[ref].numerosity
            numSum += self.popSet[ref].numerosity
        if numSum != 0:
            return sumCl/float(numSum)
        else:
            return 0

    def setIterStamps(self,currentIteration):
        for ref in self.actionSet:
            self.popSet[ref].updateTimestamp(currentIteration)

    def selectTwoParentViaTournament(self,xcs):
        selectList = [None,None]
        setList = self.actionSet

        for i in range(2):
            tSize = int(len(setList) * xcs.theta_select)
            possibleClassifiers = random.sample(setList, tSize)

            bestFitness = 0
            bestClassifier = self.actionSet[0]
            for j in possibleClassifiers:
                if self.popSet[j].fitness > bestFitness:
                    bestFitness = self.popSet[j].fitness
                    bestClassifier = j
            selectList[i] = self.popSet[bestClassifier]
        return selectList

    ####Deletion####
    def deletion(self,xcs):
        xcs.timer.startTimeDeletion()
        while (self.microPopSize > xcs.N):
            self.deleteFromPopulation(xcs)
        xcs.timer.stopTimeDeletion()

    def deleteFromPopulation(self,xcs):
        meanFitness = self.getFitnessSum()/self.microPopSize
        deletionProbSum = 0
        voteList = []
        for classifier in self.popSet:
            vote = classifier.getDelProp(meanFitness,xcs)
            deletionProbSum += vote
            voteList.append(vote)
        i = 0
        for classifier in self.popSet:
            classifier.deletionProb = voteList[i]/deletionProbSum
            i+=1

        choicePoint = deletionProbSum * random.random()
        newSum = 0
        for i in range(len(voteList)):
            classifier = self.popSet[i]
            newSum = newSum + voteList[i]
            if newSum > choicePoint:
                classifier.updateNumerosity(-1)
                self.microPopSize -= 1
                if classifier.numerosity < 1:
                    self.removeMacroClassifier(i)
                    self.deleteFromMatchSet(i)
                    self.deleteFromActionSet(i)
                    xcs.trackingObj.deletionCount += 1
                return
        return

    def getFitnessSum(self):
        sum = 0
        for classifier in self.popSet:
            sum += classifier.fitness
        return sum

    ####Clear Sets####
    def clearSets(self):
        """ Clears out references in the match and correct sets for the next learning iteration. """
        self.matchSet = []
        self.actionSet = []

    ####Evaluation####
    def makeEvaluationMatchSet(self,state,xcs):
        for i in range(len(self.popSet)):
            classifier = self.popSet[i]
            if classifier.match(state,xcs):
                self.matchSet.append(i)

    def getAveGenerality(self,xcs):
        generalitySum = 0
        for classifier in self.popSet:
            generalitySum += (xcs.env.formatData.numAttributes - len(classifier.condition))/xcs.env.formatData.numAttributes*classifier.numerosity
        if self.microPopSize == 0:
            aveGenerality = 0
        else:
            aveGenerality = generalitySum/self.microPopSize

        return aveGenerality

    def getAttributeSpecificityList(self,xcs): #To be changed for XCS
        attributeSpecList = []
        for i in range(xcs.env.formatData.numAttributes):
            attributeSpecList.append(0)
        for cl in self.popSet:
            for ref in cl.specifiedAttList:
                attributeSpecList[ref] += cl.numerosity
        return attributeSpecList

    def getAttributeAccuracyList(self,xcs): #To be changed for XCS
        attributeAccList = []
        for i in range(xcs.env.formatData.numAttributes):
            attributeAccList.append(0.0)
        for cl in self.popSet:
            for ref in cl.specifiedAttList:
                attributeAccList[ref] += cl.numerosity * cl.getAccuracy(xcs)
        return attributeAccList

