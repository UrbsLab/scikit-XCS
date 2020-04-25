
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from skXCS.Environment import Environment
from skXCS.Timer import Timer
from skXCS.ClassifierSet import ClassifierSet
from skXCS.PredictionArray import PredictionArray
from skXCS.IterationRecord import IterationRecord

import random
import numpy as np
import csv
import copy

class XCS(BaseEstimator,ClassifierMixin):
    def __init__(self,learningIterations=10000,N=1000,p_general=0.5,beta=0.15,alpha=0.1,e_0=10,nu=5,theta_GA=25,p_crossover=0.8,p_mutation=0.04,
                 theta_del=20,delta=0.1,init_prediction=10,init_e=0,init_fitness=0.01,p_explore=0.5,theta_matching=None,doGASubsumption=True,
                 doActionSetSubsumption=True,maxPayoff=1000,theta_sub=20,theta_select=0.5,discreteAttributeLimit=10,specifiedAttributes=np.array([]),
                 randomSeed="none",predictionErrorReduction=0.25,fitnessReduction=0.1,trackingFrequency=0,evalWhileFit=False):

                '''
                :param learningIterations:          The number of explore or exploit learning iterations to run
                :param N:                           Maximum micropopulation size
                :param p_general:                   Probability of generalizing an allele during covering
                :param beta:                        Learning Rate for updating statistics
                :param alpha:                       The fall of rate in the fitness evaluation
                :param e_0:                         The error threshold under which accuracy of a classifier can be set to 1
                :param nu:                          Power parameter for fitness evaluation
                :param theta_GA:                    The threshold for the GA application in an action set.
                :param p_crossover:                 The probability of applying crossover in an offspring classifier.
                :param p_mutation:                  The probability of mutating one allele and the action in an offspring classifier.
                :param theta_del:                   Specified the threshold over which the fitness of a classifier may be considered in its deletion probability.
                :param delta:                       The fraction of the mean fitness of the population below which the fitness of a classifier may be considered in its vote for deletion.
                :param init_prediction:             The initial prediction value when generating a new classifier (e.g in covering).
                :param init_e:                      The initial prediction error value when generating a new classifier (e.g in covering).
                :param init_fitness:                The initial prediction value when generating a new classifier (e.g in covering).
                :param p_explore:                   Probability of doing an explore cycle instead of an exploit cycle
                :param theta_matching:              Number of unique actions that must be represented in the match set (otherwise, covering)
                :param doGASubsumption:             Do subsumption in GA
                :param doActionSetSubsumption:      Do subsumption in [A]
                :param maxPayoff:                   For single step problems, what the maximum reward for correctness
                :param theta_sub:                   The experience of a classifier required to be a subsumer.
                :param theta_select:                The fraction of the action set to be included in tournament selection.
                :param discreteAttributeLimit:      Multipurpose param. If it is a nonnegative integer, discreteAttributeLimit determines the threshold that determines
                                                    if an attribute will be treated as a continuous or discrete attribute. For example, if discreteAttributeLimit == 10, if an attribute has more than 10 unique
                                                    values in the dataset, the attribute will be continuous. If the attribute has 10 or less unique values, it will be discrete. Alternatively,
                                                    discreteAttributeLimit can take the value of "c" or "d". See next param for this
                :param specifiedAttributes:         If "c", attributes specified by index in this param will be continuous and the rest will be discrete. If "d", attributes specified by index in this
                                                    param will be discrete and the rest will be continuous.
                :param randomSeed:                  Set a constant random seed value to some integer (in order to obtain reproducible results). Put 'none' if none (for pseudo-random algorithm runs)
                :param predictionErrorReduction:    The reduction of the prediction error when generating an offspring classifier.
                :param fitnessReduction:            The reduction of the fitness when generating an offspring classifier.
                :param trackingFrequency:           Relevant only if evalWhileFit param is true. Conducts accuracy approximations and population measurements every trackingFrequency iterations.
                                                    If param == 0, tracking done once every epoch.
                :param evalWhileFit:                Determines if live tracking and evaluation is done during model training
                '''

                # randomSeed
                if randomSeed != "none":
                    try:
                        if not self.checkIsInt(randomSeed):
                            raise Exception("randomSeed param must be integer or 'none'")
                        random.seed(int(randomSeed))
                        np.random.seed(int(randomSeed))
                    except:
                        raise Exception("randomSeed param must be integer or 'none'")

                self.learningIterations = learningIterations
                self.N = N
                self.p_general = p_general
                self.beta = beta
                self.alpha = alpha
                self.e_0 = e_0
                self.nu = nu
                self.theta_GA = theta_GA
                self.p_crossover = p_crossover
                self.p_mutation = p_mutation
                self.theta_del = theta_del
                self.delta = delta
                self.init_prediction = init_prediction
                self.init_e = init_e
                self.init_fitness = init_fitness
                self.p_explore = p_explore
                self.theta_matching = theta_matching
                self.doGASubsumption = doGASubsumption
                self.doActionSetSubsumption = doActionSetSubsumption
                self.maxPayoff = maxPayoff
                self.theta_sub = theta_sub
                self.theta_select = theta_select
                self.discreteAttributeLimit = discreteAttributeLimit
                self.specifiedAttributes = specifiedAttributes
                self.randomSeed = randomSeed
                self.predictionErrorReduction = predictionErrorReduction
                self.fitnessReduction = fitnessReduction
                self.trackingFrequency = trackingFrequency
                self.evalWhileFit = evalWhileFit

                self.hasTrained = False
                self.trackingObj = tempTrackingObj()
                self.record = IterationRecord()

    def checkIsInt(self, num):
        try:
            n = float(num)
            if num - int(num) == 0:
                return True
            else:
                return False
        except:
            return False

    ##*************** Fit ****************
    def fit(self,X,y):
        """Scikit-learn required: Supervised training of eLCS
            Parameters
            X: array-like {n_samples, n_features} Training instances. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
            y: array-like {n_samples} Training labels. ALL INSTANCE PHENOTYPES MUST BE NUMERIC NOT NAN OR OTHER TYPE
            Returns self
        """
        # If trained already, raise Exception
        if self.hasTrained:
            raise Exception("Cannot train already trained model again")

        # Check if X and Y are numeric
        try:
            for instance in X:
                for value in instance:
                    if not (np.isnan(value)):
                        float(value)
            for value in y:
                float(value)

        except:
            raise Exception("X and y must be fully numeric")

        self.env = Environment(X,y,self)

        if self.theta_matching == None:
            self.theta_matching = self.env.formatData.numberOfActions
        if self.trackingFrequency == 0:
            self.trackingFrequency = self.env.formatData.numTrainInstances

        self.timer = Timer()
        self.population = ClassifierSet(self)
        self.iterationCount = 0
        self.numCorrectGuessesDuringExploit = 0
        self.numExploitCycles = 0
        trackedAccuracy = 0
        aveGenerality = 0

        while self.iterationCount < self.learningIterations:
            state = self.env.getTrainState()
            self.runIteration(state)

            #Basic Evaluation
            self.timer.updateGlobalTimer()
            self.timer.startTimeEvaluation()
            if self.iterationCount%self.trackingFrequency == (self.trackingFrequency-1):
                if self.evalWhileFit:
                    aveGenerality = self.population.getAveGenerality(self)
                if self.numExploitCycles != 0:
                    trackedAccuracy = self.numCorrectGuessesDuringExploit/self.numExploitCycles
                self.numCorrectGuessesDuringExploit = 0
                self.numExploitCycles = 0
            self.record.addToTracking(self.iterationCount,trackedAccuracy,aveGenerality,
                                      self.trackingObj.macroPopSize,self.trackingObj.microPopSize,
                                      self.trackingObj.matchSetSize, self.trackingObj.actionSetSize,
                                      self.trackingObj.avgIterAge, self.trackingObj.subsumptionCount,
                                      self.trackingObj.crossOverCount, self.trackingObj.mutationCount,
                                      self.trackingObj.coveringCount, self.trackingObj.deletionCount,
                                      self.timer.globalTime, self.timer.globalMatching,
                                      self.timer.globalDeletion, self.timer.globalSubsumption,
                                      self.timer.globalGA, self.timer.globalEvaluation)
            self.timer.stopTimeEvaluation()
            ###

            self.iterationCount += 1
            self.env.newInstance()

        self.hasTrained = True
        return self

    def runIteration(self,state):
        self.trackingObj.resetAll()
        shouldExplore = random.random() < self.p_explore
        if shouldExplore:
            self.population.createMatchSet(state,self)
            predictionArray = PredictionArray(self.population,self)
            actionWinner = predictionArray.randomActionWinner()
            self.population.createActionSet(actionWinner)
            reward = self.env.executeAction(actionWinner)
            self.population.updateActionSet(reward,self)
            self.population.runGA(state,self)
            self.population.deletion(self)
        else:
            self.population.createMatchSet(state, self)
            predictionArray = PredictionArray(self.population, self)
            actionWinner = predictionArray.bestActionWinner()
            self.population.createActionSet(actionWinner)
            reward = self.env.executeAction(actionWinner)
            self.population.updateActionSet(reward, self)
            self.population.deletion(self)

            if reward == self.maxPayoff:
                self.numCorrectGuessesDuringExploit += 1
            self.numExploitCycles += 1

        self.trackingObj.avgIterAge = self.iterationCount - self.population.getInitStampAverage()
        self.trackingObj.macroPopSize = len(self.population.popSet)
        self.trackingObj.microPopSize = self.population.microPopSize
        self.trackingObj.matchSetSize = len(self.population.matchSet)
        self.trackingObj.actionSetSize = len(self.population.actionSet)
        self.population.clearSets()

    ##*************** Predict and Score ****************
    def predict(self,X):
        """Scikit-learn required: Test Accuracy of eLCS
            Parameters
            X: array-like {n_samples, n_features} Test instances to classify. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC
            Returns
            y: array-like {n_samples} Classifications.
        """
        try:
            for instance in X:
                for value in instance:
                    if not (np.isnan(value)):
                        float(value)
        except:
            raise Exception("X must be fully numeric")

        numInstances = X.shape[0]
        predictionList = []
        for instance in range(numInstances):
            state = X[instance]
            self.population.makeEvaluationMatchSet(state,self)
            predictionArray = PredictionArray(self.population, self)
            actionWinner = predictionArray.bestActionWinner()
            predictionList.append(actionWinner)
            self.population.clearSets()
        return np.array(predictionList)

    def predict_proba(self,X):
        """Scikit-learn required: Test Accuracy of eLCS
            Parameters
            X: array-like {n_samples, n_features} Test instances to classify. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC
            Returns
            y: array-like {n_samples} Classifications.
        """
        try:
            for instance in X:
                for value in instance:
                    if not (np.isnan(value)):
                        float(value)
        except:
            raise Exception("X must be fully numeric")

        numInstances = X.shape[0]
        predictionList = []
        for instance in range(numInstances):
            state = X[instance]
            self.population.makeEvaluationMatchSet(state, self)
            predictionArray = PredictionArray(self.population, self)
            probabilities = predictionArray.getProbabilities()
            predictionList.append(probabilities)
            self.population.clearSets()
        return np.array(predictionList)

    def score(self,X,y):
        predictionList = self.predict(X)
        return balanced_accuracy_score(y,predictionList)

    ##*************** Export and Evaluation ****************
    def exportIterationTrackingDataToCSV(self,filename='iterationData.csv'):
        if self.hasTrained:
            self.record.exportTrackingToCSV(filename)
        else:
            raise Exception("There is no tracking data to export, as the eLCS model has not been trained")

    def exportFinalRulePopulation(self,filename='rulePopulation.csv',headerNames=np.array([]),className="Action"):
        if self.hasTrained:
            numAttributes = self.env.formatData.numAttributes
            headerNames = headerNames.tolist()  # Convert to Python List

            # Default headerNames if none provided
            if len(headerNames) == 0:
                for i in range(numAttributes):
                    headerNames.append("N" + str(i))

            if len(headerNames) != numAttributes:
                raise Exception("# of Header Names provided does not match the number of attributes in dataset instances.")

            with open(filename, mode='w') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                writer.writerow(headerNames + [className] + ["Fitness","Prediction","Prediction Error","Accuracy", "Numerosity", "Avg Action Set Size",
                                                             "TimeStamp GA", "Iteration Initialized", "Specificity",
                                                             "Deletion Probability", "Experience", "Match Count"])

                classifiers = copy.deepcopy(self.population.popSet)
                for classifier in classifiers:
                    a = []
                    for attributeIndex in range(numAttributes):
                        if attributeIndex in classifier.specifiedAttList:
                            specifiedLocation = classifier.specifiedAttList.index(attributeIndex)
                            if not isinstance(classifier.condition[specifiedLocation], list):  # if discrete
                                a.append(classifier.condition[specifiedLocation])
                            else:  # if continuous
                                conditionCont = classifier.condition[specifiedLocation]  # cont array [min,max]
                                s = str(conditionCont[0]) + "," + str(conditionCont[1])
                                a.append(s)
                        else:
                            a.append("#")

                    a.append(classifier.action)
                    a.append(classifier.fitness)
                    a.append(classifier.prediction)
                    a.append(classifier.predictionError)
                    a.append(classifier.accuracy)
                    a.append(classifier.numerosity)
                    a.append(classifier.actionSetSize)
                    a.append(classifier.timeStampGA)
                    a.append(classifier.initTimeStamp)
                    a.append(len(classifier.specifiedAttList) / numAttributes)
                    a.append(classifier.deletionProb)
                    a.append(classifier.experience)
                    a.append(classifier.matchCount)
                    writer.writerow(a)
        else:
            raise Exception("There is no rule population to export, as the eLCS model has not been trained")

    def exportFinalRulePopulationDCAL(self,filename='rulePopulationDCAL.csv',headerNames=np.array([]),className="Action"):
        if self.hasTrained:
            numAttributes = self.env.formatData.numAttributes

            headerNames = headerNames.tolist()  # Convert to Python List

            # Default headerNames if none provided
            if len(headerNames) == 0:
                for i in range(numAttributes):
                    headerNames.append("N" + str(i))

            if len(headerNames) != numAttributes:
                raise Exception(
                    "# of Header Names provided does not match the number of attributes in dataset instances.")

            with open(filename, mode='w') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                writer.writerow(
                    ["Specified Values", "Specified Attribute Names"] + [className] + ["Fitness","Prediction",
                                                                                       "Prediction Error","Accuracy",
                                                                                       "Numerosity", "Avg Action Set Size",
                                                                                       "TimeStamp GA", "Iteration Initialized",
                                                                                       "Specificity", "Deletion Probability",
                                                                                       "Experience", "Match Count"])

                classifiers = copy.deepcopy(self.population.popSet)
                for classifier in classifiers:
                    a = []

                    # Add attribute information
                    headerString = ""
                    valueString = ""
                    for attributeIndex in range(numAttributes):
                        if attributeIndex in classifier.specifiedAttList:
                            specifiedLocation = classifier.specifiedAttList.index(attributeIndex)
                            headerString += str(headerNames[attributeIndex]) + ", "
                            if not isinstance(classifier.condition[specifiedLocation], list):  # if discrete
                                valueString += str(classifier.condition[specifiedLocation]) + ", "
                            else:  # if continuous
                                conditionCont = classifier.condition[specifiedLocation]  # cont array [min,max]
                                s = "[" + str(conditionCont[0]) + "," + str(conditionCont[1]) + "]"
                                valueString += s + ", "

                    a.append(valueString[:-2])
                    a.append(headerString[:-2])
                    a.append(classifier.action)

                    # Add statistics
                    a.append(classifier.action)
                    a.append(classifier.fitness)
                    a.append(classifier.prediction)
                    a.append(classifier.predictionError)
                    a.append(classifier.accuracy)
                    a.append(classifier.numerosity)
                    a.append(classifier.actionSetSize)
                    a.append(classifier.timeStampGA)
                    a.append(classifier.initTimeStamp)
                    a.append(len(classifier.specifiedAttList) / numAttributes)
                    a.append(classifier.deletionProb)
                    a.append(classifier.experience)
                    a.append(classifier.matchCount)
                    writer.writerow(a)
        else:
            raise Exception("There is no rule population to export, as the eLCS model has not been trained")

    def getFinalTrainingAccuracy(self):
        if self.hasTrained:
            originalTrainingData = self.env.formatData.savedRawTrainingData
            return self.score(originalTrainingData[0],originalTrainingData[1])
        else:
            raise Exception("There is no final training accuracy to return, as the eLCS model has not been trained")

    def getFinalInstanceCoverage(self):
        if self.hasTrained:
            numCovered = 0
            originalTrainingData = self.env.formatData.savedRawTrainingData
            for instance in originalTrainingData[0]:
                state = originalTrainingData[0][instance]
                self.population.makeEvaluationMatchSet(state, self)
                predictionArray = PredictionArray(self.population, self)
                if predictionArray.hasMatch:
                    numCovered += 1
            return numCovered/len(originalTrainingData[0])
        else:
            raise Exception("There is no final instance coverage to return, as the eLCS model has not been trained")

    def getFinalAttributeSpecificityList(self):
        if self.hasTrained:
            return self.population.getAttributeSpecificityList()
        else:
            raise Exception("There is no final attribute specificity list to return, as the eLCS model has not been trained")

    def getFinalAttributeAccuracyList(self):
        if self.hasTrained:
            return self.population.getAttributeAccuracyList()
        else:
            raise Exception("There is no final attribute accuracy list to return, as the eLCS model has not been trained")

class tempTrackingObj():
    #Tracks stats of every iteration (except accuracy, avg generality, and times)
    def __init__(self):
        self.macroPopSize = 0
        self.microPopSize = 0
        self.matchSetSize = 0
        self.correctSetSize = 0
        self.avgIterAge = 0
        self.subsumptionCount = 0
        self.crossOverCount = 0
        self.mutationCount = 0
        self.coveringCount = 0
        self.deletionCount = 0

    def resetAll(self):
        self.macroPopSize = 0
        self.microPopSize = 0
        self.matchSetSize = 0
        self.correctSetSize = 0
        self.avgIterAge = 0
        self.subsumptionCount = 0
        self.crossOverCount = 0
        self.mutationCount = 0
        self.coveringCount = 0
        self.deletionCount = 0