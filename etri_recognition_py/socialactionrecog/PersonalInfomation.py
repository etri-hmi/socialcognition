import copy
from scipy import spatial
import pickle


def CompareFeature(feature1, feature2):
    return 1 - spatial.distance.cosine(feature1, feature2)


class PersonalInformation:
    def __init__(self):
        self.nTendency = [0, 0, 0]  # passive, neutral, active
        self.nHabitCNT = [0 for _ in range(15)]

        self.nEnrolledAutoFeatureCNT = 0
        self.fAutoEnrollThreshold = 0.6
        self.bExistEnrolledFeature = False
        self.fAutoFeature = [[]]

        self.sID = ""
        self.fVerificationThreshold = 0.7
        self.fEnrolledFeature = []

        self.nUpdateCNT = 0


    def bUpdateTendency(self, nTendecny):
        if not ((nTendecny == 0) | (nTendecny == 1) | (nTendecny == 2)):
            return False

        self.nTendency[nTendecny] = self.nTendency[nTendecny] + 1
        return True

    def bUpdateHabit(self, nAction):

        self.nHabitCNT[nAction] = self.nHabitCNT[nAction] + 1
        return True

    def bCheckEnroll(self, feature):
        fCenterFeature = [0.0 for _ in range(256)]
        if self.bExistEnrolledFeature == True:
            fCenterFeature = copy.copy(self.fEnrolledFeature)
        else:
            if self.nEnrolledAutoFeatureCNT > 0:
                for ii in range(256):
                    fVal = 0.0
                    for jj in range(self.nEnrolledAutoFeatureCNT):
                        fVal = fVal + self.fAutoFeature[jj][ii]
                    fCenterFeature[ii] = fVal / float(self.nEnrolledAutoFeatureCNT)

        fSim = CompareFeature(fCenterFeature, feature)
        if fSim > self.fVerificationThreshold:
            return True
        else:
            return False

    def bEnrollFeature(self, sEnrollID, feature):
        self.sID = sEnrollID
        self.fEnrolledFeature = copy.copy(feature)
        self.bExistEnrolledFeature = True
        return True

    def bSetAutoEnrollThreshold(self, fTh):
        self.fAutoEnrollThreshold = fTh
        return True

    def bUpdateAutoFeature(self, feature):
        if self.nEnrolledAutoFeatureCNT == 0:
            self.fAutoFeature = []
            self.fAutoFeature.append(feature)
            self.nEnrolledAutoFeatureCNT = self.nEnrolledAutoFeatureCNT + 1
            self.nUpdateCNT = self.nUpdateCNT + 1
            return True

        fCenterFeature = [0.0 for _ in range(256)]
        if self.bExistEnrolledFeature == True:
            fCenterFeature = copy.copy(self.fEnrolledFeature)
        else:
            if self.nEnrolledAutoFeatureCNT > 0:
                for ii in range(256):
                    fVal = 0.0
                    for jj in range(self.nEnrolledAutoFeatureCNT):
                        fVal = fVal + self.fAutoFeature[jj][ii]
                    fCenterFeature[ii] = fVal / float(self.nEnrolledAutoFeatureCNT)

        fCenterNFeature = CompareFeature(fCenterFeature, feature)
        if fCenterNFeature < self.fVerificationThreshold:
            return False

        if self.nEnrolledAutoFeatureCNT < 3:
            self.fAutoFeature.append(feature)
            self.nEnrolledAutoFeatureCNT = self.nEnrolledAutoFeatureCNT + 1
        else:
            nMinIndex = -1
            fMinScore = 999.
            for ii in range(self.nEnrolledAutoFeatureCNT):
                fVal = CompareFeature(self.fAutoFeature[ii], feature)
                if fMinScore > fVal > self.fAutoEnrollThreshold:
                    nMinIndex = ii
                    fMinScore = fVal
            if fMinScore < fCenterNFeature:
                self.fAutoFeature[nMinIndex] = copy.copy(feature)
        self.nUpdateCNT = self.nUpdateCNT + 1
        return True


def readPersonalInformation(db_path):
    with open(db_path, "rb") as read_file:
        info = pickle.load(read_file)

    return info

def writePersonalInformation(list_personalInformation, db_path):
    with open(db_path, "wb") as write_file:
        pickle.dump(list_personalInformation, write_file)

    return True

def showPersonalInformation(personalInformation):
    print("\n\nID : {sID} "
          .format(sID=personalInformation.sID))
    print("\tExistEnrolledFeature : {bExistEnroll}"
          .format(bExistEnroll=personalInformation.bExistEnrolledFeature))
    print("\tAuto Enrolled CNT : {nCNT}".format(nCNT=personalInformation.nEnrolledAutoFeatureCNT))

    print("\tTendency : {list_Tendency}".format(list_Tendency=personalInformation.nTendency))
    # print(list_personalInformation[ii].nTendency)

    print("\tHabit CNT : {list_habit}".format(list_habit=personalInformation.nHabitCNT))
    print("\tUpdate CNT : {update_CNT}".format(update_CNT=personalInformation.nUpdateCNT))
    # print(list_personalInformation[ii].nHabitCNT)

def showPersonalInformationAll(list_personalInformation):
    nList_Size = len(list_personalInformation)
    print("\n\nList size : {list_size}".format(list_size=nList_Size))

    for ii in range(nList_Size):
        print("ID : {sID} ({iterIndex}/{totalCNT})"
              .format(sID=list_personalInformation[ii].sID, iterIndex=ii, totalCNT=nList_Size))
        print("\tExistEnrolledFeature : {bExistEnroll}"
              .format(bExistEnroll=list_personalInformation[ii].bExistEnrolledFeature))
        print("\tAuto Enrolled CNT : {nCNT}".format(nCNT=list_personalInformation[ii].nEnrolledAutoFeatureCNT))

        print("\tTendency : {list_Tendency}".format(list_Tendency=list_personalInformation[ii].nTendency))
        # print(list_personalInformation[ii].nTendency)

        print("\tHabit CNT : {list_habit}".format(list_habit=list_personalInformation[ii].nHabitCNT))
        print("\tUpdate CNT : {update_CNT}".format(update_CNT=list_personalInformation[ii].nUpdateCNT))
        # print(list_personalInformation[ii].nHabitCNT)

def autoRemoveRarePerson(list_personalInformation):
    nUpdateThreshold = 10

    nList_Size = len(list_personalInformation)

    nNeedtoDelete = []
    for ii in range(nList_Size):
        if list_personalInformation[ii].nUpdateCNT < nUpdateThreshold:
            nNeedtoDelete.append(ii)
    nNeedtoDelete.sort(reverse=True)
    for ii in range(len(nNeedtoDelete)):
        del list_personalInformation[nNeedtoDelete[ii]]






















