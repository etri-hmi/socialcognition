import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import math
import copy
from light_cnn import LightCNN_Action_Net

from openpose import pyopenpose as op

from scipy.spatial.distance import euclidean



# ========= Action Classification params =============
nNumAction = 15
# nNumJoint = 25 + 21 + 21
nNumJoint = 8 + 21 + 21

nViewFrame = 5
nCheckFrame = nViewFrame * 3
fActionArr = []
fActionProb = []

sAction = ["bitenail", "covermouth", "fighting", "fingerheart", "fingerok", "foldarms"
    ,"neutral", "pickear", "restchin", "scratch", "shakehand", "thumbchuck", "touchnose", "waving", "bowing"]
sTendency = [ "Active", "Neutral", "Passive" ]

vAllX = []
vAllY = []
vLHX = []
vLHY = []
vRHX = []
vRHY = []

fVelocityX = 0.
fVelocityY = 0.
# ========= Action Classification params =============

# ========= Openpose params =============
params = dict()
params["model_folder"] = "./models/"
params["face"] = False
params["hand"] = True

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
# ========= Openpose params =============

# ========= Tendency params =============
avgNeutralJointX = [322.14026178524216, 323.3281960896681
    , 277.6280570269355, 265.22208367646823, 261.6198277323165
    , 369.14238240576293, 381.8753715723246, 384.12814252086275
    , 383.8150782772658, 377.16965215767823, 372.35474501571855, 371.4983903382251, 371.42920561757785
    , 382.2229180099679, 378.82329372475, 374.7004049838305, 372.06152992247354
    , 386.23766577882594, 382.318045646779, 377.9146361243536, 374.59103746959545
    , 388.5917371759107, 385.05130484878816, 380.5643692650183, 377.4295040421212
    , 389.13456373253734, 386.456162121497, 383.45871614932923, 381.25423902040205
    , 261.06534784463366, 266.6972531766876, 270.331130736252, 270.8570231111664, 271.0723326401262
    , 260.0929566000788, 264.0953283027428, 268.5716093939732, 271.32327342868064
    , 257.1165693398627, 261.6365024253106, 266.3998209939861, 269.5787362852653
    , 255.62472866798808, 259.84006482310855, 264.52176403682466, 267.37121442783734
    , 256.07010035496364, 259.17439073593306, 262.3886682265787, 264.53931257396704]
avgNeutralJointY = [187.05623015997642, 250.92758932685655
    , 251.01828426255068, 326.396158656766, 398.44751644240466
    , 250.45998458556156, 325.98409536873675, 396.96939004778125
    , 399.4718000294363, 405.37457221026983, 415.9773905655077, 426.3001862842175, 433.4880929239434
    , 424.5302855937776, 435.2456918539938, 438.14686639701415, 439.4700757764309
    , 425.23371987919376, 435.28300817741234, 437.4852645445141, 437.91386246124137
    , 424.20721414015026, 432.6158489755624, 434.9105444245196, 435.39410204690677
    , 422.34410844644253, 429.5531373514435, 431.4419222423643, 432.0382540928608
    , 400.9096647112953, 406.9963011366081, 417.58202193347114, 427.61835087832156, 434.67334775218734
    , 425.13964785421615, 435.42590991313597, 438.08185959366193, 439.2206478706609
    , 425.47184133060625, 435.3077795402021, 436.7916575994253, 436.6014509559256
    , 424.5252443643856, 432.7842847999346, 434.2857559218688, 433.88741732793784
    , 422.80452726493945, 429.91303461321115, 431.1844964654373, 431.10141673363245]


# ========= Tendency params =============




def getOpenposeSkeletons(cvImg):
    datum = op.Datum()
    imageToProcess = cvImg
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])


    return datum


def convertInputJointFormat(opDatum):

    poseKeypoint = opDatum.poseKeypoints
    lhKeypoint = opDatum.handKeypoints[0]
    rhKeypoint = opDatum.handKeypoints[1]

    vInputJointX = []
    vInputJointY = []

    try:
        nCenterDif = 999
        nCenterIndex = -1
        for ii in range(len(poseKeypoint)):
            if math.fabs(320. - poseKeypoint[ii][0][0]) < nCenterDif:
                nCenterDif = math.fabs(320. - poseKeypoint[ii][0][0])
                nCenterIndex = ii


        for ii in range(8):
            vInputJointX.append(poseKeypoint[nCenterIndex][ii][0])
            vInputJointY.append(poseKeypoint[nCenterIndex][ii][1])

        for ii in range(len(lhKeypoint[nCenterIndex])):
            vInputJointX.append(lhKeypoint[nCenterIndex][ii][0])
            vInputJointY.append(lhKeypoint[nCenterIndex][ii][1])

        for ii in range(len(rhKeypoint[nCenterIndex])):
            vInputJointX.append(rhKeypoint[nCenterIndex][ii][0])
            vInputJointY.append(rhKeypoint[nCenterIndex][ii][1])

        if nCenterIndex != -1:
            return vInputJointX, vInputJointY, poseKeypoint[nCenterIndex][0][0], poseKeypoint[nCenterIndex][0][1]
        else:
            return vInputJointX, vInputJointY, 0, 0
    except Exception as e:
        print(e)
        return vInputJointX, vInputJointY, 0, 0


# updateVector
def updateJoint(vInputJointX, vInputJointY):
    global vAllX, vAllY, vLHX, vLHY, vRHX, vRHY, fVelocityX, fVelocityY

    vAllX = vAllX + vInputJointX
    vAllY = vAllY + vInputJointY

    lhx = vInputJointX[8:8+21]
    lhy = vInputJointY[8:8+21]
    rhx = vInputJointX[8+21:]
    rhy = vInputJointY[8+21:]

    vLHX = vLHX + lhx
    vLHY = vLHY + lhy
    vRHX = vRHX + rhx
    vRHY = vRHY + rhy


    if len(vAllX) > nViewFrame * nNumJoint:
        del vAllX[0:nNumJoint]
        del vAllY[0:nNumJoint]
        del vLHX[0:21]
        del vLHY[0:21]
        del vRHX[0:21]
        del vRHY[0:21]

        fVelocityX = 0.
        fVelocityY = 0.
        for ii in range(nViewFrame-1):
            if vLHX[21 * (ii + 1) + 0] != 0 and vLHX[21 * ii + 0] != 0:
                fVelocityX = fVelocityX + math.fabs(vLHX[21 * (ii + 1) + 0] - vLHX[21 * ii + 0])
            if vLHY[21 * (ii + 1) + 0] != 0 and vLHY[21 * ii + 0] != 0:
                fVelocityY = fVelocityY + math.fabs(vLHY[21 * (ii + 1) + 0] - vLHY[21 * ii + 0])



def convertToActionArr():
    global nNumJoint, nViewFrame, vAllX, vAllY, vLHX, vLHY, vRHX, vRHY
    mTransformed = np.zeros((nNumJoint+21+21, nViewFrame, 3), dtype="uint8")

    if len(vAllX) < nViewFrame*nNumJoint:
        mTransformed = cv2.resize(mTransformed, (128,128))
        return mTransformed

    tX = copy.copy(vAllX)
    tY = copy.copy(vAllY)
    tLHX = copy.copy(vLHX)
    tLHY = copy.copy(vLHY)
    tRHX = copy.copy(vRHX)
    tRHY = copy.copy(vRHY)

    tX.sort()
    tY.sort()
    tLHX.sort()
    tLHY.sort()
    tRHX.sort()
    tRHY.sort()

    while 0 in tX:
        tX.remove(0)
    while 0 in tY:
        tY.remove(0)
    while 0 in tLHX:
        tLHX.remove(0)
    while 0 in tLHY:
        tLHY.remove(0)
    while 0 in tRHX:
        tRHX.remove(0)
    while 0 in tRHY:
        tRHY.remove(0)


    for ff in range(nViewFrame):
        for jj in range(nNumJoint):
            mTransformed[jj][ff][0] = 255 * (vAllX[ff * nNumJoint + jj] - tX[0]) / (tX[len(tX) - 1] - tX[0])
            mTransformed[jj][ff][1] = 255 * (vAllY[ff * nNumJoint + jj] - tY[0]) / (tY[len(tY) - 1] - tY[0])

        if len(tLHX) >2 and len(tLHY)>2:
            nRowsOffset = nNumJoint
            for jj in range(21):
                mTransformed[jj + nRowsOffset][ff][0] = 255 * (vLHX[ff * 21 + jj] - tLHX[0]) / (tLHX[len(tLHX) - 1] - tLHX[0])
                mTransformed[jj + nRowsOffset][ff][1] = 255 * (vLHY[ff * 21 + jj] - tLHY[0]) / (tLHY[len(tLHY) - 1] - tLHY[0])
        if len(tRHX) >2 and len(tRHY)>2:
            nRowsOffset = nNumJoint + 21
            for jj in range(21):
                mTransformed[jj + nRowsOffset][ff][0] = 255 * (vRHX[ff * 21 + jj] - tRHX[0]) / (tRHX[len(tRHX) - 1] - tRHX[0])
                mTransformed[jj + nRowsOffset][ff][1] = 255 * (vRHY[ff * 21 + jj] - tRHY[0]) / (tRHY[len(tRHY) - 1] - tRHY[0])


    mTransformed = cv2.resize(mTransformed, (128,128))
    return mTransformed


def updateAction(nAction):
    global fActionArr, nCheckFrame
    for ii in range(nCheckFrame-1, 0, -1):
        fActionArr[ii] = fActionArr[ii-1]
    fActionArr[0] = nAction

def getTopNAction(nTopN, convertedImg):
    global fActionProb, fActionArr, nCheckFrame, nNumAction, sAction

    fActionRank = []

    if nTopN > nNumAction:
        fActionRank.append((-1, -1))
        return fActionRank, "nTopN is out of scope."

    if (convertedImg[80:90, 0:128] == 0).all() or (convertedImg[118:128, 0:128] == 0).all():
        fActionRank.append((-2, -2))
        return fActionRank, "Hands not detected."

    fActionProb = [0 for _ in range(nNumAction)]
    fTemp = [0 for _ in range(nNumAction)]

    for ii in range(nCheckFrame):
        fActionProb[fActionArr[ii]] = fActionProb[fActionArr[ii]] + 1
    fSum = 0.

    for ii in range(nNumAction):
        fExp = math.exp(fActionProb[ii])
        fSum = fSum + fExp
        fTemp[ii] = fExp

    for ii in range(nNumAction):
        fActionRank[ii] = (fTemp[ii] / fSum, ii)

    fActionRank.sort(reverse=True)

    sTopN = ""
    for ii in range(nTopN):
        sActionNProb = "{sAction} : {fProb:0.1f} \n".format(sAction=sAction[fActionRank[ii][1]]
                                                         , fProb=fActionRank[ii][0]*100)
        sTopN  = sTopN + sActionNProb

    # fActionRank는 확률과 행동id를 포함한 튜플로 구성된 리스트(높은확률부터 내림차순 정렬).
    # 최근 nCheckFrame번의 인식 결과에 기반하여 확률계산.
    # 즉, 이번 프레임에서 인식 된 행동의 확률 리스트는 아니라는 의미.
    # 메세지로 전달할 때는 fActionRank[0]을 사용
    return fActionRank, sTopN




def EAR_Initialization(path):
    global fActionArr, nCheckFrame
    fActionArr = [0 for _ in range(nCheckFrame)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(path, 'ETRI_BODYACTION.pth.tar')
    EAR_Net = LightCNN_Action_Net(num_classes=14)
    EAR_Net = torch.nn.DataParallel(EAR_Net).to(device)
    EAR_Net.load_state_dict(torch.load(model_path)['state_dict'])
    EAR_Net.eval()

    return EAR_Net

def EAR_BodyAction_Estimation(EAR_Net, convertedImg):
    global fVelocityX, fVelocityY, vAllX, vAllY


    # bowing
    fShoulder = math.fabs(vAllX[nNumJoint * (nViewFrame - 1) + 5] - vAllX[nNumJoint * (nViewFrame - 1) + 2])
    fNeck = math.fabs(vAllY[nNumJoint*(nViewFrame-1) + 0] - vAllY[nNumJoint*(nViewFrame-1) + 1])
    if fNeck < fShoulder/6 or vAllY[nNumJoint*(nViewFrame-1) + 0] > vAllY[nNumJoint*(nViewFrame-1) + 1]:
        return 14

    else:
        convertedImg = cv2.cvtColor(convertedImg, cv2.COLOR_BGR2RGB)
        img_trim_in = convertedImg / 255.
        img_trim_in = np.transpose(img_trim_in, axes=[2, 0, 1])
        img_trim_in = np.array(img_trim_in, dtype=np.float32)
        img_trim_in = torch.from_numpy(img_trim_in)
        img_trim_in = torch.unsqueeze(img_trim_in, 0)
        output = EAR_Net(img_trim_in)
        output_cpu = output.cpu()
        output_np = output_cpu.detach().numpy().squeeze().tolist()

        return output_np.index(max(output_np))


def drawJoint(cvImg, vInputJointX, vInputJointY):
    xLen = len(vInputJointX)
    yLen = len(vInputJointY)

    if not xLen == yLen:
        return cvImg

    for ii in range(xLen):
        cv2.circle(cvImg, (vInputJointX[ii], vInputJointY[ii]), 2, (0,255,0), -1)

    return cvImg


def euc_dist(pt1,pt2):
    return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))

def alignSkeleton():
    global vAllX, vAllY, avgNeutralJointX, avgNeutralJointY, nNumJoint, nViewFrame

    alignedNeutralX = []
    alignedNeutralY = []

    for ff in range(nViewFrame):
        copyNeutralX = copy.deepcopy(avgNeutralJointX)
        copyNeutralY = copy.deepcopy(avgNeutralJointY)
        frameX = vAllX[ff*nNumJoint + 0:ff*nNumJoint + nNumJoint]
        frameY = vAllY[ff*nNumJoint + 0:ff*nNumJoint + nNumJoint]

        # Scale
        fActionJoint01D = euc_dist((frameX[0],frameY[0]), (frameX[1],frameY[1]))
        fAvgJoint10D = euc_dist((copyNeutralX[0],copyNeutralY[0]), (copyNeutralX[1],copyNeutralY[1]))
        fScale = fActionJoint01D / fAvgJoint10D
        for ii in range(len(copyNeutralX)):
            # copyNeutralX[ii] = copyNeutralX[ii] * fScale
            copyNeutralY[ii] = copyNeutralY[ii] * fScale

        # translation
        fXOffset = frameX[1] - copyNeutralX[1]
        fYOffset = frameY[1] - copyNeutralY[1]
        # for ii in range(len(copyNeutralX)):
        #     copyNeutralX[ii] = copyNeutralX[ii] + fXOffset
        #     copyNeutralY[ii] = copyNeutralY[ii] + fYOffset
        # for ii in range(len(copyNeutralX)):
        copyNeutralX = copyNeutralX + fXOffset
        copyNeutralY = copyNeutralY + fYOffset

        alignedNeutralX = alignedNeutralX + np.ndarray.tolist(copyNeutralX)
        alignedNeutralY = alignedNeutralY + np.ndarray.tolist(copyNeutralY)

    return alignedNeutralX, alignedNeutralY


def getVectorDistance(alignedNeutralX, alignedNeutralY):
    global vAllX, vAllY, nNumJoint, nViewFrame

    if len(alignedNeutralX) != len(vAllX) or len(alignedNeutralY) != len(vAllY):
        return -1

    distSum = 0.

    for ii in range(len(vAllX)):
        if vAllX[ii] == 0:
            alignedNeutralX[ii] = 0
        if vAllY[ii] == 0:
            alignedNeutralY[ii] = 0

    dist = euclidean(vAllX, alignedNeutralX)
    distSum = distSum + dist
    dist = euclidean(vAllY, alignedNeutralY)
    distSum = distSum + dist

    return distSum


def getTendencyCategory(fDistance):
    if fDistance<1500:
        return 1
    elif fDistance<4000:
        return 0
    else:
        return 2






















