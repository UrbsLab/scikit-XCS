
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
        actionsNotCovered = copy.deepcopy(xcs.env.formatData.phenotypeList)

        matchSetSize = 0
        for i in range(len(self.popSet)):
            classifier = self.popSet[i]
            if classifier.match(state,xcs):
                self.matchSet.append(i)
                matchSetSize+=classifier.numerosity
                if classifier.action in actionsNotCovered:
                    actionsNotCovered.remove(classifier.action)

        if xcs.env.formatData.isBinaryClassification:
            doCovering = len(actionsNotCovered) != 0 or len(self.matchSet) < 5 #Second condition only holds for 1 covering round
        else:
            doCovering = len(actionsNotCovered) != 0

        while doCovering:
            action = random.choice(actionsNotCovered)
            coveredClassifier = Classifier(self)
            coveredClassifier.initializeWithMatchingStateAndGivenAction(matchSetSize,state,action,xcs)
            self.addClassifierToPopulation(xcs,coveredClassifier,True)
            self.matchSet.append(len(self.popSet)-1)

            doCovering = len(actionsNotCovered) != 0

    def getIdenticalClassifier(self,xcs,newClassifier):
        for classifier in self.popSet:
            if newClassifier.equals(xcs,classifier):
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
        if xcs.doActionSetSubsumption:
            self.doActionSetSubsumption(xcs)

    def updateFitnessSet(self,xcs):
        accuracySum = 0
        accuracies = []

        i = 0
        for clRef in self.actionSet:
            classifier = self.popSet[clRef]
            accuracies.append(classifier.getAccuracy())
            accuracySum = accuracySum + accuracies[i]*classifier.numerosity
            i+=1

        for clRef in self.actionSet:
            classifier = self.popSet[clRef]
            classifier.updateFitness(accuracySum,accuracies[i],xcs)

    ####Action Set Subsumption####
    def doActionSetSubsumption(self,xcs):
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



