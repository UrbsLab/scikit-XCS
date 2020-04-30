import csv
import numpy as np

class IterationRecord():
    '''
    IterationRecord Tracks 1 dictionary:
    1) Tracking Dict: Cursory Iteration Evaluation. Frequency determined by trackingFrequency param in eLCS. For each iteration evaluated, it saves:
        KEY-iteration number
        0-accuracy (approximate from correct array in eLCS)
        1-average population generality
        2-macropopulation size
        3-micropopulation size
        4-match set size
        5-correct set size
        6-average iteration age of action set classifiers
        7-number of classifiers subsumed (in iteration)
        8-number of crossover operations performed (in iteration)
        9-number of mutation operations performed (in iteration)
        10-number of covering operations performed (in iteration)
        11-number of deleted macroclassifiers performed (in iteration)
        12-total global time at end of iteration
        13-total matching time at end of iteration
        14-total deletion time at end of iteration
        15-total subsumption time at end of iteration
        16-total selection time at end of iteration
        17-total evaluation time at end of iteration
    '''

    def __init__(self):
        self.trackingDict = {}

    def addToTracking(self,iterationNumber,accuracy,avgPopGenerality,macroSize,microSize,mSize,aSize,iterAvg,
                      subsumptionCount,crossoverCount,mutationCount,coveringCount,deletionCount,
                      globalTime,matchingTime,deletionTime,subsumptionTime,gaTime,evaluationTime):

        self.trackingDict[iterationNumber] = [accuracy,avgPopGenerality,macroSize,microSize,mSize,aSize,iterAvg,
                                   subsumptionCount,crossoverCount,mutationCount,coveringCount,deletionCount,
                                   globalTime,matchingTime,deletionTime,subsumptionTime,gaTime,evaluationTime]

    def exportTrackingToCSV(self,filename='iterationData.csv'):
        #Exports each entry in Tracking Array as a column
        with open(filename,mode='w') as file:
            writer = csv.writer(file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)

            writer.writerow(["Iteration","Accuracy (approx)", "Average Population Generality","Macropopulation Size",
                             "Micropopulation Size", "Match Set Size", "Action Set Size", "Average Iteration Age of Action Set Classifiers",
                             "# Classifiers Subsumed in Iteration","# Crossover Operations Performed in Iteration","# Mutation Operations Performed in Iteration",
                             "# Covering Operations Performed in Iteration","# Deletion Operations Performed in Iteration",
                             "Total Global Time","Total Matching Time","Total Deletion Time","Total Subsumption Time","Total GA Time","Total Evaluation Time"])

            for k,v in sorted(self.trackingDict.items()):
                writer.writerow([k,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17]])
        file.close()