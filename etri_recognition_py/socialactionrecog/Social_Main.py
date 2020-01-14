#!/usr/bin/env python
#-*- encoding: utf8 -*-
import cv2
import PersonalInfomation as PI
import ETRI_Action_Recognition as EAR

import copy
import os


def main():
    # 행동 정보 인식 모델 초기화
    EAR_Net = EAR.EAR_Initialization()


    # opencv Cam Interface 초기화
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 개인 정보 관리를 위해 기존 데이터 존재여부 확인
    nFrameCNT = 0
    nTopNAction = 1
    if os.path.exists("./PersonalData/DB.DAT"):
        list_PI = PI.readPersonalInformation()
        PI.autoRemoveRarePerson(list_PI)
        print("%d person load Done." % len(list_PI))

    else:
        list_PI = []


    bReady = True
    if vc.isOpened() == False:
        bReady = False
        print("Error : Cam is not opened.")


    # 인식 프로세스 시작
    while(bReady):
        # 얼굴 검출 정보 / 행동인식 정보 메모리 초기화
        list_ETRIFace = []
        nMainFaceIndex = -1
        bUpdatePI = True
        nAction = -1
        nTendency = -1

        ret, frame = vc.read()
        if ret:
            img_show = copy.deepcopy(frame)

            # Action Recognition
            #######################
            # Get Skeleton data
            opDatum = EAR.getOpenposeSkeletons(frame)
            # Get Joint / Center Face Position
            vInputJointX, vInputJointY, nCenterFaceX, nCenterFaceY = EAR.convertInputJointFormat(opDatum)

            # skeleton 제대로 들어왓는지 필터링
            if vInputJointX != [] and vInputJointY != []:
                # for Action ===================================================
                # 행동인식기에 joint 정보 업데이트
                EAR.updateJoint(vInputJointX, vInputJointY)
                # joint 정보를 2D Map으로 변환
                convertedImg = EAR.convertToActionArr()

                # draw convertedImg & skeleton to frame
                img_show = EAR.drawJoint(img_show, vInputJointX, vInputJointY)
                img_show = cv2.flip(img_show, 1)
                img_show[0:128, 512:640] = copy.copy(convertedImg)

                # joint 정보가 ViewFrame 만큼 쌓인 경우에만 행동인식 수행
                if len(EAR.vAllX) == EAR.nViewFrame * EAR.nNumJoint:
                    # 행동 종류 인식
                    nAction = EAR.EAR_BodyAction_Estimation(EAR_Net, convertedImg)
                    # print("Action : %s" % EAR.sAction[nAction])

                    # 행동 정보 업데이트
                    EAR.updateAction(nAction)

                    # 출력을 위해 Top N 행동 종류와 확률을 string으로 받기
                    sActionResult = EAR.getTopNAction(nTopNAction, convertedImg)
                    cv2.putText(img_show, sActionResult, (15, 40), 0, 0.7, (0, 0, 255), 2)

                # for State Update ===============================================
                if bUpdatePI and nAction != -1:
                    # for Tendency ===================================================
                    # skeleton 위치 정규화
                    a, b = EAR.alignSkeleton()
                    # skeleton 움직임 기반 Tendency score 계산
                    fTendencyScore = EAR.getVectorDistance(a, b)
                    # Score 기반 성향 계산
                    nTendency = EAR.getTendencyCategory(fTendencyScore)


                # 개인 정보에 행동 및 태도 정보 업데이트.
                if nAction != -1:
                    list_PI[nMainFaceIndex].bUpdateHabit(nAction)
                if nTendency != -1:
                    list_PI[nMainFaceIndex].bUpdateTendency(nTendency)


            # Personal Information Update
            #######################
            # 기술이전 등록 기술 - 미포함


            # 인식 결과 출력
            cv2.imshow("TT", img_show)
            nKey = cv2.waitKey(1)

            # esc로 업데이트 된 개인정보 저장하고 프로그램 종료
            if nKey == 27:
                PI.writePersonalInformation(list_PI)
                break
            # "I" 키로 등록된 개인정보 리스트 전체 확인
            elif nKey == 73 or nKey == 105:
                PI.showPersonalInformationAll(list_PI)
            # "R" 키로 부정확하게 등록된 정보 삭제하고 남은 리스트 확인
            elif nKey == 82 or nKey == 114 :
                PI.autoRemoveRarePerson(list_PI)
                PI.showPersonalInformationAll(list_PI)

            nFrameCNT = nFrameCNT + 1




if __name__ == "__main__":
    main()

