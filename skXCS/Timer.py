# Import Required Modules---------------
import time


# --------------------------------------

class Timer:
    def __init__(self):
        # Global Time objects
        self.globalStartRef = time.time()
        self.globalTime = 0.0
        self.globalAdd = 0

        # Match Time Variables
        self.startRefMatching = 0.0
        self.globalMatching = 0.0

        # Deletion Time Variables
        self.startRefDeletion = 0.0
        self.globalDeletion = 0.0

        # Subsumption Time Variables
        self.startRefSubsumption = 0.0
        self.globalSubsumption = 0.0

        # GA Time Variables
        self.startRefGA = 0.0
        self.globalGA = 0.0

        # Evaluation Time Variables
        self.startRefEvaluation = 0.0
        self.globalEvaluation = 0.0

        # ************************************************************

    def startTimeMatching(self):
        """ Tracks MatchSet Time """
        self.startRefMatching = time.time()

    def stopTimeMatching(self):
        """ Tracks MatchSet Time """
        diff = time.time() - self.startRefMatching
        self.globalMatching += diff

        # ************************************************************

    def startTimeDeletion(self):
        """ Tracks Deletion Time """
        self.startRefDeletion = time.time()

    def stopTimeDeletion(self):
        """ Tracks Deletion Time """
        diff = time.time() - self.startRefDeletion
        self.globalDeletion += diff

    # ************************************************************
    def startTimeSubsumption(self):
        """Tracks Subsumption Time """
        self.startRefSubsumption = time.time()

    def stopTimeSubsumption(self):
        """Tracks Subsumption Time """
        diff = time.time() - self.startRefSubsumption
        self.globalSubsumption += diff

        # ************************************************************

    def startTimeGA(self):
        """ Tracks Selection Time """
        self.startRefGA = time.time()

    def stopTimeGA(self):
        """ Tracks Selection Time """
        diff = time.time() - self.startRefGA
        self.globalGA += diff

    # ************************************************************
    def startTimeEvaluation(self):
        """ Tracks Evaluation Time """
        self.startRefEvaluation = time.time()

    def stopTimeEvaluation(self):
        """ Tracks Evaluation Time """
        diff = time.time() - self.startRefEvaluation
        self.globalEvaluation += diff

        # ************************************************************

    def updateGlobalTimer(self):
        """ Set the global end timer, call at very end of algorithm. """
        self.globalTime = (time.time() - self.globalStartRef) + self.globalAdd
        return self.globalTime
