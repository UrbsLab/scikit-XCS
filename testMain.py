

import pandas as pd
from skXCS.XCS import XCS
from skXCS.StringEnumerator import StringEnumerator
import random
import numpy as np
from sklearn.model_selection import cross_val_score

converter = StringEnumerator("test/DataSets/Real/Multiplexer11Modified.csv","Class")
headers,classLabel,dataFeatures,dataActions = converter.getParams()

#Shuffle Data Before CV
formatted = np.insert(dataFeatures,dataFeatures.shape[1],dataActions,1)
np.random.shuffle(formatted)
dataFeatures = np.delete(formatted,-1,axis=1)
dataActions = formatted[:,-1]

model = XCS(learningIterations=10000,N=1000)
#print(np.mean(cross_val_score(model,dataFeatures,dataActions,cv=3)))

model.fit(dataFeatures,dataActions)
print(model.score(dataFeatures,dataActions))
model.exportIterationTrackingDataToCSV("defaultExportDir/tracking.csv")
model.exportFinalRulePopulation("defaultExportDir/rulePop.csv",headers,classLabel)
model.exportFinalRulePopulationDCAL("defaultExportDir/rulePopDCAL.csv",headers,classLabel)
print(model.predict_proba(dataFeatures))
print(model.getFinalInstanceCoverage())
print(model.getFinalTrainingAccuracy())
print(model.getFinalAttributeAccuracyList())
print(model.getFinalAttributeSpecificityList())