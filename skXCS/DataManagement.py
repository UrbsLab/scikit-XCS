
import numpy as np

class DataManagement:
    def __init__(self,X,y,xcs):
        self.savedRawTrainingData = [X,y]
        self.numAttributes = X.shape[1]
        self.attributeInfoType = [0] * self.numAttributes             # stores false (d) or true (c) depending on its type, which points to parallel reference in one of the below 2 arrays
        self.attributeInfoContinuous = [[0, 0]] * self.numAttributes  # stores continuous ranges and NaN otherwise
        self.attributeInfoDiscrete = [0] * self.numAttributes         # stores arrays of discrete values or NaN otherwise.
        for i in range(0,self.numAttributes):
            self.attributeInfoDiscrete[i] = AttributeInfoDiscreteElement()
        self.discretePhenotype = True
        self.phenotypeList = []  # Stores all possible discrete phenotype values

        self.isDefault = True  # Is discrete attribute limit an int or string
        try:
            int(xcs.discrete_attribute_limit)
        except:
            self.isDefault = False

        self.numTrainInstances = X.shape[0]  # The number of instances in the training data
        self.discriminateClasses(y)
        self.isBinaryClassification = len(self.phenotypeList) == 2
        self.numberOfActions = len(self.phenotypeList)

        self.discriminateAttributes(X, xcs)
        self.characterizeAttributes(X)
        self.trainFormatted = self.formatData(X, y)

    def discriminateClasses(self,phenotypes):
        currentPhenotypeIndex = 0
        classCount = {}
        while (currentPhenotypeIndex < self.numTrainInstances):
            target = phenotypes[currentPhenotypeIndex]
            if target in self.phenotypeList:
                classCount[target]+=1
            else:
                self.phenotypeList.append(target)
                classCount[target] = 1
            currentPhenotypeIndex+=1

    def discriminateAttributes(self,features,xcs):
        for att in range(self.numAttributes):
            attIsDiscrete = True
            if self.isDefault:
                currentInstanceIndex = 0
                stateDict = {}
                while attIsDiscrete and len(list(stateDict.keys())) <= xcs.discrete_attribute_limit and currentInstanceIndex < self.numTrainInstances:
                    target = features[currentInstanceIndex,att]
                    if target in list(stateDict.keys()):
                        stateDict[target] += 1
                    elif np.isnan(target):
                        pass
                    else:
                        stateDict[target] = 1
                    currentInstanceIndex+=1

                if len(list(stateDict.keys())) > xcs.discrete_attribute_limit:
                    attIsDiscrete = False
            elif xcs.discrete_attribute_limit == "c":
                if att in xcs.specified_attributes:
                    attIsDiscrete = False
                else:
                    attIsDiscrete = True
            elif xcs.discrete_attribute_limit == "d":
                if att in xcs.specified_attributes:
                    attIsDiscrete = True
                else:
                    attIsDiscrete = False

            if attIsDiscrete:
                self.attributeInfoType[att] = False
            else:
                self.attributeInfoType[att] = True


    def characterizeAttributes(self,features):
        for currentFeatureIndexInAttributeInfo in range(self.numAttributes):
            for currentInstanceIndex in range(self.numTrainInstances):
                target = features[currentInstanceIndex,currentFeatureIndexInAttributeInfo]
                if not self.attributeInfoType[currentFeatureIndexInAttributeInfo]:#if attribute is discrete
                    if target in self.attributeInfoDiscrete[currentFeatureIndexInAttributeInfo].distinctValues or np.isnan(target):
                        pass
                    else:
                        self.attributeInfoDiscrete[currentFeatureIndexInAttributeInfo].distinctValues.append(target)
                else: #if attribute is continuous
                    if np.isnan(target):
                        pass
                    elif float(target) > self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][1]:
                        self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][1] = float(target)
                    elif float(target) < self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][0]:
                        self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][0] = float(target)
                    else:
                        pass

    def formatData(self,features,phenotypes):
        formatted = np.insert(features,self.numAttributes,phenotypes,1) #Combines features and phenotypes into one array
        np.random.shuffle(formatted)
        shuffledFeatures = formatted[:,:-1].tolist()
        shuffledLabels = formatted[:,self.numAttributes].tolist()
        for i in range(len(shuffledFeatures)):
            for j in range(len(shuffledFeatures[i])):
                if np.isnan(shuffledFeatures[i][j]):
                    shuffledFeatures[i][j] = None
            if np.isnan(shuffledLabels[i]):
                shuffledLabels[i] = None
        return [shuffledFeatures,shuffledLabels]


class AttributeInfoDiscreteElement():
    def __init__(self):
        self.distinctValues = []