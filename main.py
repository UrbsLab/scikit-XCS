

import pandas as pd
from skXCS.XCS import XCS
from skXCS.StringEnumerator import StringEnumerator
import random
import numpy as np
from sklearn.model_selection import cross_val_score

converter = StringEnumerator("test/DataSets/Real/Multiplexer11Modified.csv","Class")
headers,classLabel,dataFeatures,dataActions = converter.getParams()

model = XCS(learningIterations=1500)
model.fit(dataFeatures,dataActions)
print(model.score(dataFeatures,dataActions))
model.exportFinalRulePopulation("defaultExportDir/rulePop.csv",headers,classLabel)
model.exportIterationTrackingData('defaultExportDir/tracking.csv')
model.pickleModel('defaultExportDir/pickled')

model2 = XCS(learningIterations=1500,rebootFilename='defaultExportDir/pickled')
model2.fit(dataFeatures,dataActions)
print(model2.score(dataFeatures,dataActions))
model2.exportFinalRulePopulation("defaultExportDir/rulePop2.csv",headers,classLabel)
model2.exportIterationTrackingData('defaultExportDir/tracking2.csv')
model2.pickleModel('defaultExportDir/pickled2')

model3 = XCS(learningIterations=2000,rebootFilename='defaultExportDir/pickled2')
model3.fit(dataFeatures,dataActions)
print(model3.score(dataFeatures,dataActions))
model3.exportFinalRulePopulation("defaultExportDir/rulePop3.csv",headers,classLabel)
model3.exportIterationTrackingData('defaultExportDir/tracking3.csv')



# print(model.predict_proba(dataFeatures))
# print(model.getFinalInstanceCoverage())
# print(model.getFinalTrainingAccuracy())
# print(model.getFinalAttributeAccuracyList())
# print(model.getFinalAttributeSpecificityList())