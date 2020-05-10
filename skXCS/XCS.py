
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
import pickle
import time

class XCS(BaseEstimator,ClassifierMixin):
    def __init__(self,learning_iterations=10000,N=1000,p_general=0.5,beta=0.2,alpha=0.1,e_0=10,nu=5,theta_GA=25,p_crossover=0.8,p_mutation=0.04,
                 theta_del=20,delta=0.1,init_prediction=10,init_e=0,init_fitness=0.01,p_explore=0.5,theta_matching=None,do_GA_subsumption=True,
                 do_action_set_subsumption=False,max_payoff=1000,theta_sub=20,theta_select=0.5,discrete_attribute_limit=10,specified_attributes=np.array([]),
                 random_state=None,prediction_error_reduction=0.25,fitness_reduction=0.1,reboot_filename=None):

                '''
                :param learning_iterations:          Must be nonnegative integer. The number of explore or exploit learning iterations to run
                :param N:                           Must be nonnegative integer. Maximum micropopulation size
                :param p_general:                   Must be float from 0 - 1. Probability of generalizing an allele during covering
                :param beta:                        Must be float. Learning Rate for updating statistics
                :param alpha:                       Must be float. The fall of rate in the fitness evaluation
                :param e_0:                         Must be float. The error threshold under which accuracy of a classifier can be set to 1
                :param nu:                          Must be float. Power parameter for fitness evaluation
                :param theta_GA:                    Must be nonnegative float. The threshold for the GA application in an action set
                :param p_crossover:                 Must be float from 0 - 1. The probability of applying crossover in an offspring classifier
                :param p_mutation:                  Must be float from 0 - 1. The probability of mutating one allele and the action in an offspring classifier
                :param theta_del:                   Must be nonnegative integer. Specified the threshold over which the fitness of a classifier may be considered in its deletion probability
                :param delta:                       Must be float. The fraction of the mean fitness of the population below which the fitness of a classifier may be considered in its vote for deletion
                :param init_prediction:             Must be float. The initial prediction value when generating a new classifier (e.g in covering)
                :param init_e:                      Must be float. The initial prediction error value when generating a new classifier (e.g in covering)
                :param init_fitness:                Must be float. The initial prediction value when generating a new classifier (e.g in covering)
                :param p_explore:                   Must be float from 0 - 1. Probability of doing an explore cycle instead of an exploit cycle
                :param theta_matching:              Must be nonnegative integer. Number of unique actions that must be represented in the match set (otherwise, covering)
                :param do_GA_subsumption:             Must be boolean. Do subsumption in GA
                :param do_action_set_subsumption:      Must be boolean. Do subsumption in [A]
                :param max_payoff:                   Must be float. For single step problems, what the maximum reward for correctness
                :param theta_sub:                   Must be nonnegative integer. The experience of a classifier required to be a subsumer
                :param theta_select:                Must be float from 0 - 1. The fraction of the action set to be included in tournament selection
                :param discrete_attribute_limit:      Must be nonnegative integer OR "c" OR "d". Multipurpose param. If it is a nonnegative integer, discrete_attribute_limit determines the threshold that determines
                                                    if an attribute will be treated as a continuous or discrete attribute. For example, if discrete_attribute_limit == 10, if an attribute has more than 10 unique
                                                    values in the dataset, the attribute will be continuous. If the attribute has 10 or less unique values, it will be discrete. Alternatively,
                                                    discrete_attribute_limit can take the value of "c" or "d". See next param for this
                :param specified_attributes:         Must be an ndarray type of nonnegative integer attributeIndices (zero indexed).
                                                    If "c", attributes specified by index in this param will be continuous and the rest will be discrete. If "d", attributes specified by index in this
                                                    param will be discrete and the rest will be continuous.
                :param random_state:                  Must be an integer or None. Set a constant random seed value to some integer (in order to obtain reproducible results). Put None if none (for pseudo-random algorithm runs)
                :param prediction_error_reduction:    Must be float. The reduction of the prediction error when generating an offspring classifier
                :param fitness_reduction:            Must be float. The reduction of the fitness when generating an offspring classifier
                :param reboot_filename:    Must be String or None. Filename of model to be rebooted
                '''

                #learning_iterations
                if not self.checkIsInt(learning_iterations):
                    raise Exception("learning_iterations param must be nonnegative integer")

                if learning_iterations < 0:
                    raise Exception("learning_iterations param must be nonnegative integer")

                #N
                if not self.checkIsInt(N):
                    raise Exception("N param must be nonnegative integer")

                if N < 0:
                    raise Exception("N param must be nonnegative integer")

                #p_general
                if not self.checkIsFloat(p_general):
                    raise Exception("p_general param must be float from 0 - 1")

                if p_general < 0 or p_general > 1:
                    raise Exception("p_general param must be float from 0 - 1")

                #beta
                if not self.checkIsFloat(beta):
                    raise Exception("beta param must be float")

                #alpha
                if not self.checkIsFloat(alpha):
                    raise Exception("alpha param must be float")

                #e_0
                if not self.checkIsFloat(e_0):
                    raise Exception("e_0 param must be float")

                #nu
                if not self.checkIsFloat(nu):
                    raise Exception("nu param must be float")

                #theta_GA
                if not self.checkIsFloat(theta_GA):
                    raise Exception("theta_GA param must be nonnegative float")

                if theta_GA < 0:
                    raise Exception("theta_GA param must be nonnegative float")

                #p_crossover
                if not self.checkIsFloat(p_crossover):
                    raise Exception("p_crossover param must be float from 0 - 1")

                if p_crossover < 0 or p_crossover > 1:
                    raise Exception("p_crossover param must be float from 0 - 1")

                #p_mutation
                if not self.checkIsFloat(p_mutation):
                    raise Exception("p_mutation param must be float from 0 - 1")

                if p_mutation < 0 or p_mutation > 1:
                    raise Exception("p_mutation param must be float from 0 - 1")

                #theta_del
                if not self.checkIsInt(theta_del):
                    raise Exception("theta_del param must be nonnegative integer")

                if theta_del < 0:
                    raise Exception("theta_del param must be nonnegative integer")

                #delta
                if not self.checkIsFloat(delta):
                    raise Exception("delta param must be float")

                #init_prediction
                if not self.checkIsFloat(init_prediction):
                    raise Exception("init_prediction param must be float")

                #init_e
                if not self.checkIsFloat(init_e):
                    raise Exception("init_e param must be float")

                #init_fitness
                if not self.checkIsFloat(init_fitness):
                    raise Exception("init_fitness param must be float")

                #p_explore
                if not self.checkIsFloat(p_explore):
                    raise Exception("p_explore param must be float from 0 - 1")

                if p_explore < 0 or p_explore > 1:
                    raise Exception("p_explore param must be float from 0 - 1")

                #theta_matching
                if not self.checkIsInt(theta_matching) and theta_matching != None:
                    raise Exception("theta_matching param must be nonnegative integer")

                if theta_matching != None and theta_matching < 0:
                    raise Exception("theta_matching param must be nonnegative integer")

                #do_GA_subsumption
                if not (isinstance(do_GA_subsumption, bool)):
                    raise Exception("do_GA_subsumption param must be boolean")

                #do_action_set_subsumption
                if not (isinstance(do_action_set_subsumption, bool)):
                    raise Exception("do_action_set_subsumption param must be boolean")

                #max_payoff
                if not self.checkIsFloat(max_payoff):
                    raise Exception("max_payoff param must be float")

                #theta_sub
                if not self.checkIsInt(theta_sub):
                    raise Exception("theta_sub param must be nonnegative integer")

                if theta_sub < 0:
                    raise Exception("theta_sub param must be nonnegative integer")

                #theta_select
                if not self.checkIsFloat(theta_select):
                    raise Exception("theta_select param must be float from 0 - 1")

                if theta_select < 0 or theta_select > 1:
                    raise Exception("theta_select param must be float from 0 - 1")

                #discrete_attribute_limit
                if discrete_attribute_limit != "c" and discrete_attribute_limit != "d":
                    try:
                        dpl = int(discrete_attribute_limit)
                        if not self.checkIsInt(discrete_attribute_limit):
                            raise Exception("discrete_attribute_limit param must be nonnegative integer or 'c' or 'd'")
                        if dpl < 0:
                            raise Exception("discrete_attribute_limit param must be nonnegative integer or 'c' or 'd'")
                    except:
                        raise Exception("discrete_attribute_limit param must be nonnegative integer or 'c' or 'd'")

                #specified_attributes
                if not (isinstance(specified_attributes, np.ndarray)):
                    raise Exception("specified_attributes param must be ndarray")

                for spAttr in specified_attributes:
                    if not self.checkIsInt(spAttr):
                        raise Exception("All specified_attributes elements param must be nonnegative integers")
                    if int(spAttr) < 0:
                        raise Exception("All specified_attributes elements param must be nonnegative integers")

                #prediction_error_reduction
                if not self.checkIsFloat(prediction_error_reduction):
                    raise Exception("prediction_error_reduction param must be float")

                #fitness_reduction
                if not self.checkIsFloat(fitness_reduction):
                    raise Exception("fitness_reduction param must be float")

                #rebootPopulationFilename
                if reboot_filename != None and not isinstance(reboot_filename,str):
                    raise Exception("reboot_filename param must be None or String from pickle")

                # random_state
                if random_state != None:
                    try:
                        if not self.checkIsInt(random_state):
                            raise Exception("random_state param must be integer or None")
                        random.seed(int(random_state))
                        np.random.seed(int(random_state))
                    except:
                        raise Exception("random_state param must be integer or None")

                self.learning_iterations = learning_iterations
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
                self.do_GA_subsumption = do_GA_subsumption
                self.do_action_set_subsumption = do_action_set_subsumption
                self.max_payoff = max_payoff
                self.theta_sub = theta_sub
                self.theta_select = theta_select
                self.discrete_attribute_limit = discrete_attribute_limit
                self.specified_attributes = specified_attributes
                self.random_state = random_state
                self.prediction_error_reduction = prediction_error_reduction
                self.fitness_reduction = fitness_reduction

                self.hasTrained = False
                self.trackingObj = TempTrackingObj()
                self.record = IterationRecord()
                self.reboot_filename = reboot_filename

    def checkIsInt(self, num):
        try:
            n = float(num)
            if num - int(num) == 0:
                return True
            else:
                return False
        except:
            return False

    def checkIsFloat(self,num):
        try:
            n = float(num)
            return True
        except:
            return False

    ##*************** Fit ****************
    def fit(self,X,y):
        """Scikit-learn required: Supervised training of XCS
            Parameters
            X: array-like {n_samples, n_features} Training instances. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
            y: array-like {n_samples} Training labels. ALL INSTANCE PHENOTYPES MUST BE NUMERIC NOT NAN OR OTHER TYPE
            Returns self
        """

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
        if self.theta_matching > self.env.formatData.numberOfActions:
            raise Exception("theta_matching param cannot be greater than the number of actions")

        self.iterationCount = 0

        self.trackedAccuracy = []
        self.movingAvgCount = 50
        aveGenerality = 0
        aveGeneralityFreq = min(self.env.formatData.numTrainInstances,int(self.learning_iterations/20)+1)

        if self.reboot_filename == None:
            self.timer = Timer()
            self.population = ClassifierSet()
        else:
            self.rebootPopulation()

        while self.iterationCount < self.learning_iterations:
            state = self.env.getTrainState()
            self.runIteration(state)

            #Basic Evaluation
            self.timer.updateGlobalTimer()
            self.timer.startTimeEvaluation()
            if self.iterationCount%aveGeneralityFreq == (aveGeneralityFreq-1):
                aveGenerality = self.population.getAveGenerality(self)

            if len(self.trackedAccuracy) != 0:
                accuracy = sum(self.trackedAccuracy)/len(self.trackedAccuracy)
            else:
                accuracy = 0
            self.record.addToTracking(self.iterationCount,accuracy,aveGenerality,
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
        self.saveFinalMetrics()
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

            if reward == self.max_payoff:
                if len(self.trackedAccuracy) == self.movingAvgCount:
                    del self.trackedAccuracy[0]
                self.trackedAccuracy.append(1)
            else:
                if len(self.trackedAccuracy) == self.movingAvgCount:
                    del self.trackedAccuracy[0]
                self.trackedAccuracy.append(0)

        self.trackingObj.avgIterAge = self.iterationCount - self.population.getInitStampAverage()
        self.trackingObj.macroPopSize = len(self.population.popSet)
        self.trackingObj.microPopSize = self.population.microPopSize
        self.trackingObj.matchSetSize = len(self.population.matchSet)
        self.trackingObj.actionSetSize = len(self.population.actionSet)
        self.population.clearSets()

    ##*************** Population Reboot ****************
    def saveFinalMetrics(self):
        self.finalMetrics = [self.learning_iterations,self.timer.globalTime, self.timer.globalMatching,
                             self.timer.globalDeletion, self.timer.globalSubsumption, self.timer.globalGA,
                             self.timer.globalEvaluation,copy.deepcopy(self.population.popSet)]

    def pickle_model(self,filename=None):
        if self.hasTrained:
            if filename == None:
                filename = 'pickled'+str(int(time.time()))
            outfile = open(filename,'wb')
            pickle.dump(self.finalMetrics,outfile)
            outfile.close()
        else:
            raise Exception("There is model to pickle, as the XCS model has not been trained")

    def rebootPopulation(self):
        #Sets popSet and microPopSize of self.population, as well as trackingMetrics,
        file = open(self.reboot_filename,'rb')
        rawData = pickle.load(file)
        file.close()

        popSet = rawData[7]
        microPopSize = 0
        for rule in popSet:
            microPopSize += rule.numerosity
        set = ClassifierSet()
        set.popSet = popSet
        set.microPopSize = microPopSize
        self.population = set
        self.timer = Timer()
        self.timer.globalAdd = rawData[1]
        self.timer.globalMatching = rawData[2]
        self.timer.globalDeletion = rawData[3]
        self.timer.globalSubsumption = rawData[4]
        self.timer.globalGA = rawData[5]
        self.timer.globalEvaluation = rawData[6]
        self.learning_iterations += rawData[0]
        self.iterationCount += rawData[0]

    ##*************** Predict and Score ****************
    def predict(self,X):
        """Scikit-learn required: Test Accuracy of XCS
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
        """Scikit-learn required: Test Accuracy of XCS
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
    def export_iteration_tracking_data(self,filename='iterationData.csv'):
        if self.hasTrained:
            self.record.exportTrackingToCSV(filename)
        else:
            raise Exception("There is no tracking data to export, as the XCS model has not been trained")

    def export_final_rule_population(self,filename='rulePopulation.csv',headerNames=np.array([]),className="Action"):
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
                    a.append(classifier.getAccuracy(self))
                    a.append(classifier.numerosity)
                    a.append(classifier.actionSetSize)
                    a.append(classifier.timestampGA)
                    a.append(classifier.initTimeStamp)
                    a.append(len(classifier.specifiedAttList) / numAttributes)
                    a.append(classifier.deletionProb)
                    a.append(classifier.experience)
                    a.append(classifier.matchCount)
                    writer.writerow(a)
            file.close()
        else:
            raise Exception("There is no rule population to export, as the XCS model has not been trained")

    def export_final_rule_population_DCAL(self,filename='rulePopulationDCAL.csv',headerNames=np.array([]),className="Action"):
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

                    # Add statistics
                    a.append(classifier.action)
                    a.append(classifier.fitness)
                    a.append(classifier.prediction)
                    a.append(classifier.predictionError)
                    a.append(classifier.getAccuracy(self))
                    a.append(classifier.numerosity)
                    a.append(classifier.actionSetSize)
                    a.append(classifier.timestampGA)
                    a.append(classifier.initTimeStamp)
                    a.append(len(classifier.specifiedAttList) / numAttributes)
                    a.append(classifier.deletionProb)
                    a.append(classifier.experience)
                    a.append(classifier.matchCount)
                    writer.writerow(a)
            file.close()
        else:
            raise Exception("There is no rule population to export, as the XCS model has not been trained")

    def get_final_training_accuracy(self):
        if self.hasTrained:
            originalTrainingData = self.env.formatData.savedRawTrainingData
            return self.score(originalTrainingData[0],originalTrainingData[1])
        else:
            raise Exception("There is no final training accuracy to return, as the XCS model has not been trained")

    def get_final_instance_coverage(self):
        if self.hasTrained:
            numCovered = 0
            originalTrainingData = self.env.formatData.savedRawTrainingData
            for state in originalTrainingData[0]:
                self.population.makeEvaluationMatchSet(state, self)
                predictionArray = PredictionArray(self.population, self)
                if predictionArray.hasMatch:
                    numCovered += 1
                self.population.clearSets()
            return numCovered/len(originalTrainingData[0])
        else:
            raise Exception("There is no final instance coverage to return, as the XCS model has not been trained")

    def get_final_attribute_specificity_list(self):
        if self.hasTrained:
            return self.population.getAttributeSpecificityList(self)
        else:
            raise Exception("There is no final attribute specificity list to return, as the XCS model has not been trained")

    def get_final_attribute_accuracy_list(self):
        if self.hasTrained:
            return self.population.getAttributeAccuracyList(self)
        else:
            raise Exception("There is no final attribute accuracy list to return, as the XCS model has not been trained")

class TempTrackingObj():
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