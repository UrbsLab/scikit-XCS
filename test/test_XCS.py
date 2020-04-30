import unittest
import numpy as np
from skXCS.XCS import XCS
from skXCS.StringEnumerator import StringEnumerator
from sklearn.model_selection import cross_val_score
import os

THIS_DIR = os.path.dirname(os.path.abspath("test_eLCS.py"))
if THIS_DIR[-4:] == 'test': #Patch that ensures testing from Scikit not test directory
    THIS_DIR = THIS_DIR[:-5]

class test_XCS(unittest.TestCase):
    #learningIterations (nonnegative integer)
    def testParamLearningIterationsNonnumeric(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(learningIterations="hello")
        self.assertTrue("learningIterations param must be nonnegative integer" in str(context.exception))

    def testParamLearningIterationsInvalidNumeric(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(learningIterations=3.3)
        self.assertTrue("learningIterations param must be nonnegative integer" in str(context.exception))

    def testParamLearningIterationsInvalidNumeric2(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(learningIterations=-2)
        self.assertTrue("learningIterations param must be nonnegative integer" in str(context.exception))

    def testParamLearningIterations(self):
        clf = XCS(learningIterations=2000)
        self.assertEqual(clf.learningIterations,2000)

    #N (nonnegative integer)
    def testParamNNonnumeric(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(N="hello")
        self.assertTrue("N param must be nonnegative integer" in str(context.exception))

    def testParamNInvalidNumeric(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(N=3.3)
        self.assertTrue("N param must be nonnegative integer" in str(context.exception))

    def testParamNInvalidNumeric2(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(N=-2)
        self.assertTrue("N param must be nonnegative integer" in str(context.exception))

    def testParamN(self):
        clf = XCS(N=2000)
        self.assertEqual(clf.N,2000)

    #p_general (float 0-1)
    def testParamP_GeneralInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(p_general="hello")
        self.assertTrue("p_general param must be float from 0 - 1" in str(context.exception))

    def testParamP_GeneralInv2(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(p_general=3)
        self.assertTrue("p_general param must be float from 0 - 1" in str(context.exception))

    def testParamP_GeneralInv3(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(p_general=-1.2)
        self.assertTrue("p_general param must be float from 0 - 1" in str(context.exception))

    def testParamP_General1(self):
        clf = XCS(p_general=0)
        self.assertEqual(clf.p_general,0)

    def testParamP_General2(self):
        clf = XCS(p_general=0.3)
        self.assertEqual(clf.p_general,0.3)

    def testParamP_General3(self):
        clf = XCS(p_general=1)
        self.assertEqual(clf.p_general,1)

    #beta (float)
    def testBetaInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(beta="hi")
        self.assertTrue("beta param must be float" in str(context.exception))

    def testBeta1(self):
        clf = XCS(beta = -1)
        self.assertEqual(clf.beta,-1)

    def testBeta2(self):
        clf = XCS(beta = 3)
        self.assertEqual(clf.beta,3)

    def testBeta3(self):
        clf = XCS(beta = 1.2)
        self.assertEqual(clf.beta,1.2)

    #alpha (float)
    def testAlphaInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(alpha="hi")
        self.assertTrue("alpha param must be float" in str(context.exception))

    def testAlpha1(self):
        clf = XCS(alpha = -1)
        self.assertEqual(clf.alpha,-1)

    def testAlpha2(self):
        clf = XCS(alpha = 3)
        self.assertEqual(clf.alpha,3)

    def testAlpha3(self):
        clf = XCS(alpha = 1.2)
        self.assertEqual(clf.alpha,1.2)

    #e_0 (float)
    def testE0Inv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(e_0="hi")
        self.assertTrue("e_0 param must be float" in str(context.exception))

    def testE01(self):
        clf = XCS(e_0 = -1)
        self.assertEqual(clf.e_0,-1)

    def testE02(self):
        clf = XCS(e_0 = 3)
        self.assertEqual(clf.e_0,3)

    def testE03(self):
        clf = XCS(e_0 = 1.2)
        self.assertEqual(clf.e_0,1.2)

    #nu (float)
    def testNuInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(nu="hi")
        self.assertTrue("nu param must be float" in str(context.exception))

    def testNu1(self):
        clf = XCS(nu = -1)
        self.assertEqual(clf.nu,-1)

    def testNu2(self):
        clf = XCS(nu = 3)
        self.assertEqual(clf.nu,3)

    def testNu3(self):
        clf = XCS(nu = 1.2)
        self.assertEqual(clf.nu,1.2)

    #theta_GA (nonnegative float)
    def testParamThetaGAInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_GA="hello")
        self.assertTrue("theta_GA param must be nonnegative float" in str(context.exception))

    def testParamThetaGAInv3(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_GA=-1.2)
        self.assertTrue("theta_GA param must be nonnegative float" in str(context.exception))

    def testParamThetaGA1(self):
        clf = XCS(theta_GA=0)
        self.assertEqual(clf.theta_GA,0)

    def testParamThetaGA2(self):
        clf = XCS(theta_GA=1)
        self.assertEqual(clf.theta_GA,1)

    def testParamThetaGA3(self):
        clf = XCS(theta_GA=4.3)
        self.assertEqual(clf.theta_GA,4.3)

    #p_crossover (float 0-1)
    def testParamP_CrossoverInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(p_crossover="hello")
        self.assertTrue("p_crossover param must be float from 0 - 1" in str(context.exception))

    def testParamP_CrossoverInv2(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(p_crossover=3)
        self.assertTrue("p_crossover param must be float from 0 - 1" in str(context.exception))

    def testParamP_CrossoverInv3(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(p_crossover=-1.2)
        self.assertTrue("p_crossover param must be float from 0 - 1" in str(context.exception))

    def testParamP_Crossover1(self):
        clf = XCS(p_crossover=0)
        self.assertEqual(clf.p_crossover,0)

    def testParamP_Crossover2(self):
        clf = XCS(p_crossover=0.3)
        self.assertEqual(clf.p_crossover,0.3)

    def testParamP_Crossover3(self):
        clf = XCS(p_crossover=1)
        self.assertEqual(clf.p_crossover,1)

    #p_mutation (float 0-1)
    def testParamP_MutationInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(p_mutation="hello")
        self.assertTrue("p_mutation param must be float from 0 - 1" in str(context.exception))

    def testParamP_MutationInv2(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(p_mutation=3)
        self.assertTrue("p_mutation param must be float from 0 - 1" in str(context.exception))

    def testParamP_MutationInv3(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(p_mutation=-1.2)
        self.assertTrue("p_mutation param must be float from 0 - 1" in str(context.exception))

    def testParamP_Mutation1(self):
        clf = XCS(p_mutation=0)
        self.assertEqual(clf.p_mutation,0)

    def testParamP_Mutation2(self):
        clf = XCS(p_mutation=0.3)
        self.assertEqual(clf.p_mutation,0.3)

    def testParamP_Mutation3(self):
        clf = XCS(p_mutation=1)
        self.assertEqual(clf.p_mutation,1)

    #theta_del (nonnegative integer)
    def testParamThetaDelInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_del="hello")
        self.assertTrue("theta_del param must be nonnegative integer" in str(context.exception))

    def testParamThetaDelInv2(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_del=2.3)
        self.assertTrue("theta_del param must be nonnegative integer" in str(context.exception))

    def testParamThetaDelInv3(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_del=-1.2)
        self.assertTrue("theta_del param must be nonnegative integer" in str(context.exception))

    def testParamThetaDelInv4(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_del=-5)
        self.assertTrue("theta_del param must be nonnegative integer" in str(context.exception))

    def testParamThetaDel1(self):
        clf = XCS(theta_del=0)
        self.assertEqual(clf.theta_del,0)

    def testParamThetaDel2(self):
        clf = XCS(theta_del=5)
        self.assertEqual(clf.theta_del,5)

    #delta (float)
    def testDeltaInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(delta="hi")
        self.assertTrue("delta param must be float" in str(context.exception))

    def testDelta1(self):
        clf = XCS(delta = -1)
        self.assertEqual(clf.delta,-1)

    def testDelta2(self):
        clf = XCS(delta = 3)
        self.assertEqual(clf.delta,3)

    def testDelta3(self):
        clf = XCS(delta = 1.2)
        self.assertEqual(clf.delta,1.2)

    #init_predication (float)
    def testInitPredictionInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(init_prediction="hi")
        self.assertTrue("init_prediction param must be float" in str(context.exception))

    def testInitPrediction1(self):
        clf = XCS(init_prediction = -1)
        self.assertEqual(clf.init_prediction,-1)

    def testInitPrediction2(self):
        clf = XCS(init_prediction = 3)
        self.assertEqual(clf.init_prediction,3)

    def testInitPrediction3(self):
        clf = XCS(init_prediction = 1.2)
        self.assertEqual(clf.init_prediction,1.2)

    #init_e (float)
    def testInitEInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(init_e="hi")
        self.assertTrue("init_e param must be float" in str(context.exception))

    def testInitE1(self):
        clf = XCS(init_e = -1)
        self.assertEqual(clf.init_e,-1)

    def testInitE2(self):
        clf = XCS(init_e = 3)
        self.assertEqual(clf.init_e,3)

    def testInitE3(self):
        clf = XCS(init_e = 1.2)
        self.assertEqual(clf.init_e,1.2)

    #init_fitness (float)
    def testInitFitnessInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(init_fitness="hi")
        self.assertTrue("init_fitness param must be float" in str(context.exception))

    def testInitFitness1(self):
        clf = XCS(init_fitness = -1)
        self.assertEqual(clf.init_fitness,-1)

    def testInitFitness2(self):
        clf = XCS(init_fitness = 3)
        self.assertEqual(clf.init_fitness,3)

    def testInitFitness3(self):
        clf = XCS(init_fitness = 1.2)
        self.assertEqual(clf.init_fitness,1.2)

    #p_explore (float 0-1)
    def testParamP_ExploreInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(p_explore="hello")
        self.assertTrue("p_explore param must be float from 0 - 1" in str(context.exception))

    def testParamP_ExploreInv2(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(p_explore=3)
        self.assertTrue("p_explore param must be float from 0 - 1" in str(context.exception))

    def testParamP_ExploreInv3(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(p_explore=-1.2)
        self.assertTrue("p_explore param must be float from 0 - 1" in str(context.exception))

    def testParamP_Explore1(self):
        clf = XCS(p_explore=0)
        self.assertEqual(clf.p_explore,0)

    def testParamP_Explore2(self):
        clf = XCS(p_explore=0.3)
        self.assertEqual(clf.p_explore,0.3)

    def testParamP_Explore3(self):
        clf = XCS(p_explore=1)
        self.assertEqual(clf.p_explore,1)

    #theta_matching (nonnegative integer)
    def testParamThetaMatchingInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_matching="hello")
        self.assertTrue("theta_matching param must be nonnegative integer" in str(context.exception))

    def testParamThetaMatchingInv2(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_matching=2.3)
        self.assertTrue("theta_matching param must be nonnegative integer" in str(context.exception))

    def testParamThetaMatchingInv3(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_matching=-1.2)
        self.assertTrue("theta_matching param must be nonnegative integer" in str(context.exception))

    def testParamThetaMatchingInv4(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_matching=-5)
        self.assertTrue("theta_matching param must be nonnegative integer" in str(context.exception))

    def testParamThetaMatching1(self):
        clf = XCS(theta_matching=0)
        self.assertEqual(clf.theta_matching,0)

    def testParamThetaMatching2(self):
        clf = XCS(theta_matching=5)
        self.assertEqual(clf.theta_matching,5)

    #doGASubsumption (boolean)
    def testDoSub2Invalid(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(doGASubsumption=2)
        self.assertTrue("doGASubsumption param must be boolean" in str(context.exception))

    def testDoSub2(self):
        clf = XCS(doGASubsumption=True)
        self.assertEqual(clf.doGASubsumption,True)

    #doActionSetSubsumption (boolean)
    def testDoSubInvalid(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(doActionSetSubsumption=2)
        self.assertTrue("doActionSetSubsumption param must be boolean" in str(context.exception))

    def testDoSub(self):
        clf = XCS(doActionSetSubsumption=True)
        self.assertEqual(clf.doActionSetSubsumption,True)

    #maxPayoff (float)
    def testMaxPayoffInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(maxPayoff="hi")
        self.assertTrue("maxPayoff param must be float" in str(context.exception))

    def testMaxPayoff1(self):
        clf = XCS(maxPayoff = -1)
        self.assertEqual(clf.maxPayoff,-1)

    def testMaxPayoff2(self):
        clf = XCS(maxPayoff = 3)
        self.assertEqual(clf.maxPayoff,3)

    def testMaxPayoff3(self):
        clf = XCS(maxPayoff = 1.2)
        self.assertEqual(clf.maxPayoff,1.2)

    #theta_sub (nonnegative integer)
    def testParamThetaSubInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_sub="hello")
        self.assertTrue("theta_sub param must be nonnegative integer" in str(context.exception))

    def testParamThetaSubInv2(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_sub=2.3)
        self.assertTrue("theta_sub param must be nonnegative integer" in str(context.exception))

    def testParamThetaSubInv3(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_sub=-1.2)
        self.assertTrue("theta_sub param must be nonnegative integer" in str(context.exception))

    def testParamThetaSubInv4(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_sub=-5)
        self.assertTrue("theta_sub param must be nonnegative integer" in str(context.exception))

    def testParamThetaSub1(self):
        clf = XCS(theta_sub=0)
        self.assertEqual(clf.theta_sub,0)

    def testParamThetaSub2(self):
        clf = XCS(theta_sub=5)
        self.assertEqual(clf.theta_sub,5)

    #theta_select (float 0-1)
    def testParamThetaSelInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_select="hello")
        self.assertTrue("theta_select param must be float from 0 - 1" in str(context.exception))

    def testParamThetaSelInv2(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_select=3)
        self.assertTrue("theta_select param must be float from 0 - 1" in str(context.exception))

    def testParamThetaSelInv3(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(theta_select=-1.2)
        self.assertTrue("theta_select param must be float from 0 - 1" in str(context.exception))

    def testParamThetaSel1(self):
        clf = XCS(theta_select=0)
        self.assertEqual(clf.theta_select, 0)

    def testParamThetaSel2(self):
        clf = XCS(theta_select=0.3)
        self.assertEqual(clf.theta_select, 0.3)

    def testParamThetaSel3(self):
        clf = XCS(theta_select=1)
        self.assertEqual(clf.theta_select, 1)

    #discreteAttributeLimit (nonnegative integer or 'c/d'
    def testDiscreteAttributeLimitInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(discreteAttributeLimit="h")
        self.assertTrue("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'" in str(context.exception))

    def testDiscreteAttributeLimitInv2(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(discreteAttributeLimit=-10)
        self.assertTrue("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'" in str(context.exception))

    def testDiscreteAttributeLimitInv3(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(discreteAttributeLimit=1.2)
        self.assertTrue("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'" in str(context.exception))

    def testDiscreteAttributeLimit1(self):
        clf = XCS(discreteAttributeLimit=10)
        self.assertEqual(clf.discreteAttributeLimit,10)

    def testDiscreteAttributeLimit2(self):
        clf = XCS(discreteAttributeLimit="c")
        self.assertEqual(clf.discreteAttributeLimit,"c")

    def testDiscreteAttributeLimit3(self):
        clf = XCS(discreteAttributeLimit="d")
        self.assertEqual(clf.discreteAttributeLimit,"d")

    #specifiedAttributes (ndarray of nonnegative integer attributes
    def testParamSpecAttrNonarray(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(specifiedAttributes=2)
        self.assertTrue("specifiedAttributes param must be ndarray" in str(context.exception))

    def testParamSpecAttrNonnumeric(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(specifiedAttributes=np.array([2,100,"hi",200]))
        self.assertTrue("All specifiedAttributes elements param must be nonnegative integers" in str(context.exception))

    def testParamSpecAttrInvalidNumeric(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(specifiedAttributes=np.array([2,100,200.2,200]))
        self.assertTrue("All specifiedAttributes elements param must be nonnegative integers" in str(context.exception))

    def testParamSpecAttrInvalidNumeric2(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(specifiedAttributes=np.array([2,100,-200,200]))
        self.assertTrue("All specifiedAttributes elements param must be nonnegative integers" in str(context.exception))

    def testParamSpecAttr(self):
        clf = XCS(specifiedAttributes=np.array([2, 100, 200, 300]))
        self.assertTrue(np.array_equal(clf.specifiedAttributes,np.array([2, 100, 200, 300])))

    #randomSeed (integer or "none")
    def testRandomSeedInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(randomSeed="hello")
        self.assertTrue("randomSeed param must be integer or 'none'" in str(context.exception))

    def testRandomSeedInv2(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(randomSeed=1.2)
        self.assertTrue("randomSeed param must be integer or 'none'" in str(context.exception))

    def testRandomSeed2(self):
        clf = XCS(randomSeed=200)
        self.assertEqual(clf.randomSeed,200)

    def testRandomSeed3(self):
        clf = XCS(randomSeed='none')
        self.assertEqual(clf.randomSeed,'none')

    #predictionErrorReduction (float)
    def testPredReductionInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(predictionErrorReduction="hi")
        self.assertTrue("predictionErrorReduction param must be float" in str(context.exception))

    def testPredReduction1(self):
        clf = XCS(predictionErrorReduction = -1)
        self.assertEqual(clf.predictionErrorReduction,-1)

    def testPredReduction2(self):
        clf = XCS(predictionErrorReduction = 3)
        self.assertEqual(clf.predictionErrorReduction,3)

    def testPredReduction3(self):
        clf = XCS(predictionErrorReduction = 1.2)
        self.assertEqual(clf.predictionErrorReduction,1.2)

    #fitnessReduction (float)
    def testFitnessReductionInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(fitnessReduction="hi")
        self.assertTrue("fitnessReduction param must be float" in str(context.exception))

    def testFitnessReduction1(self):
        clf = XCS(fitnessReduction = -1)
        self.assertEqual(clf.fitnessReduction,-1)

    def testFitnessReduction2(self):
        clf = XCS(fitnessReduction = 3)
        self.assertEqual(clf.fitnessReduction,3)

    def testFitnessReduction3(self):
        clf = XCS(fitnessReduction = 1.2)
        self.assertEqual(clf.fitnessReduction,1.2)

    #rebootFilename (None or String)
    def testRebootFilenameInv1(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(rebootFilename=2)
        self.assertTrue("rebootFilename param must be None or String from pickle" in str(context.exception))

    def testRebootFilenameInv2(self):
        with self.assertRaises(Exception) as context:
            clf = XCS(rebootFilename=True)
        self.assertTrue("rebootFilename param must be None or String from pickle" in str(context.exception))

    def testRebootFilename1(self):
        clf = XCS()
        self.assertEqual(clf.rebootFilename,None)

    def testRebootFilename2(self):
        clf = XCS(rebootFilename=None)
        self.assertEqual(clf.rebootFilename,None)

    def testRebootFilename3(self):
        clf = XCS(rebootFilename='hello')
        self.assertEqual(clf.rebootFilename,'hello')

    #Performance Tests
    #6B MP 1000 iter training
    def test6BitMP1000Iterations(self):
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Real/Multiplexer6Modified.csv")
        converter = StringEnumerator(dataPath,"Class")
        headers, classLabel, dataFeatures, dataPhenotypes = converter.getParams()
        clf = XCS(learningIterations=1000,N=500,nu=10)
        clf.fit(dataFeatures,dataPhenotypes)
        answer = 0.894
        #print("6 Bit 1000 Iter: "+str(clf.getFinalTrainingAccuracy()))
        self.assertTrue(self.approxEqualOrBetter(0.2,clf.getFinalTrainingAccuracy(),answer,True))

    # 6B MP 5000 iter training
    def test6BitMP5000Iterations(self):
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Real/Multiplexer6Modified.csv")
        converter = StringEnumerator(dataPath, "Class")
        headers, classLabel, dataFeatures, dataPhenotypes = converter.getParams()
        clf = XCS(learningIterations=5000, N=500, nu=10)
        clf.fit(dataFeatures, dataPhenotypes)
        answer = 1
        #print("6 Bit 5000 Iter: "+str(clf.getFinalTrainingAccuracy()))
        self.assertTrue(self.approxEqualOrBetter(0.2, clf.getFinalTrainingAccuracy(), answer, True))

    #11B MP 5000 iter training
    def test11BitMP5000Iterations(self):
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Real/Multiplexer11Modified.csv")
        converter = StringEnumerator(dataPath,"Class")
        headers, classLabel, dataFeatures, dataPhenotypes = converter.getParams()
        clf = XCS(learningIterations=5000,N=1000,nu=10)
        clf.fit(dataFeatures,dataPhenotypes)
        answer = 0.9514
        #print("11 Bit 5000 Iter: "+str(clf.getFinalTrainingAccuracy()))
        self.assertTrue(self.approxEqualOrBetter(0.2,clf.getFinalTrainingAccuracy(),answer,True))

    #20B MP 5000 iter training
    def test20BitMP5000Iterations(self):
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Real/Multiplexer20Modified.csv")
        converter = StringEnumerator(dataPath,"Class")
        headers, classLabel, dataFeatures, dataPhenotypes = converter.getParams()
        clf = XCS(learningIterations=5000,N=2000,nu=10)
        clf.fit(dataFeatures,dataPhenotypes)
        answer = 0.6634
        #print("20 Bit 5000 Iter: "+str(clf.getFinalTrainingAccuracy()))
        self.assertTrue(self.approxEqualOrBetter(0.2,clf.getFinalTrainingAccuracy(),answer,True))

    #Continuous Valued 5000 iter training
    def testContValues5000Iterations(self):
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Real/ContinuousAndNonBinaryDiscreteAttributes.csv")
        converter = StringEnumerator(dataPath,"Class")
        headers, classLabel, dataFeatures, dataPhenotypes = converter.getParams()
        clf = XCS(learningIterations=5000)
        clf.fit(dataFeatures,dataPhenotypes)
        answer = 0.64
        #print("Continuous Attributes 5000 Iter: "+str(clf.getFinalTrainingAccuracy()))
        self.assertTrue(self.approxEqualOrBetter(0.2,clf.getFinalTrainingAccuracy(),answer,True))

    #3-fold testing 6B MP 1000 iter
    def test6BitMPTesting1000Iterations(self):
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Real/Multiplexer6Modified.csv")
        converter = StringEnumerator(dataPath,"Class")
        headers, classLabel, dataFeatures, dataPhenotypes = converter.getParams()
        formatted = np.insert(dataFeatures, dataFeatures.shape[1], dataPhenotypes, 1)
        np.random.shuffle(formatted)
        dataFeatures = np.delete(formatted, -1, axis=1)
        dataPhenotypes = formatted[:, -1]

        clf = XCS(learningIterations=1000,N=500,nu=10)
        score = np.mean(cross_val_score(clf, dataFeatures, dataPhenotypes, cv=3))

        answer = 0.9
        #print("6 Bit Testing 1000 Iter: "+str(score))
        self.assertTrue(self.approxEqualOrBetter(0.2,score,answer,True))

    #3-fold testing Continuous Valued + Missing 5000 iter
    def testContValuesAndMissingTesting5000Iterations(self):
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Real/ContinuousAndNonBinaryDiscreteAttributesMissing.csv")
        converter = StringEnumerator(dataPath, "Class")
        headers, classLabel, dataFeatures, dataPhenotypes = converter.getParams()
        formatted = np.insert(dataFeatures, dataFeatures.shape[1], dataPhenotypes, 1)
        np.random.shuffle(formatted)
        dataFeatures = np.delete(formatted, -1, axis=1)
        dataPhenotypes = formatted[:, -1]

        clf = XCS(learningIterations=5000)
        score = np.mean(cross_val_score(clf, dataFeatures, dataPhenotypes, cv=3))

        answer = 0.5
        #print("Cont & Missing Testing 5000 Iter: " + str(score))
        self.assertTrue(self.approxEqualOrBetter(0.2, score, answer, True))

    #Random Seed Testing - Done

    #Reboot Pop Testing -

    ###Util Functions###
    def approxEqual(self, threshold, comp, right):  # threshold is % tolerance
        return abs(abs(comp - right) / right) < threshold

    def approxEqualOrBetter(self, threshold, comp, right,
                            better):  # better is False when better is less, True when better is greater
        if not better:
            if self.approxEqual(threshold, comp, right) or comp < right:
                return True
            return False
        else:
            if self.approxEqual(threshold, comp, right) or comp > right:
                return True
            return False
