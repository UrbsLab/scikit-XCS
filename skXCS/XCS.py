
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score

import numpy as np

class XCS(BaseEstimator,ClassifierMixin):
    def __init__(self,learningIterations=10000,N=1000,p_general=0.5,beta=0.15,alpha=0.1,e_0=10,nu=5,theta_GA=25,p_crossover=0.8,p_mutation=0.04,
                 theta_del=20,delta=0.1,init_prediction=10,init_e=0,init_fitness=0.01,p_explore=0.5,theta_matching=None,doGASubsumption=True,
                 doActionSetSubsumption=True,maxPayoff=1000,theta_sub=20,theta_select=0.5,discreteAttributeLimit=10,specifiedAttributes=np.array([]),
                 randomSeed="none",predictionErrorReduction=0.25,fitnessReduction=0.1):

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
                '''

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

                self.hasTrained = False

    def fit(self,X,y):
        """Scikit-learn required: Supervised training of eLCS

            Parameters
            ----------
            X: array-like {n_samples, n_features}
                Training instances. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
            y: array-like {n_samples}
                Training labels. ALL INSTANCE PHENOTYPES MUST BE NUMERIC NOT NAN OR OTHER TYPE

            Returns
            __________
            self
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

