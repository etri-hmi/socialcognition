#!/usr/bin/env python
#-*- encoding: utf8 -*-

'''
ETRI Social Behavior Recognition ROS Node

Author: Minsu Jang (minsu@etri.re.kr)
'''


import os
import json
import time
import sys

import cv2
import numpy as np
import torch
import copy

import rospy
import rospkg
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError


pkg_path = rospkg.RosPack().get_path('etri_recognition_py')
core_path = os.path.join(pkg_path, 'socialactionrecog')
try:
    sys.path.index(core_path) # Or os.getcwd() for this directory
except ValueError:
    sys.path.append(core_path) # Or os.getcwd() for this directory

import ETRI_Face_Information as EFI
import PersonalInfomation as PI
import ETRI_Action_Recognition as EAR


class ETRISocialBehaviorRecognitionNode(object):
    def __init__(self):
        super(ETRISocialBehaviorRecognitionNode, self).__init__()

        rospy.loginfo('[%s] Initializing...', rospy.get_name())

        self.model_path = os.path.join(core_path, 'models')
        self.person_data_path = os.path.join(core_path, 'PersonData')
        self.enrolled_data_path = os.path.join(core_path, 'EnrolledData')
        self.person_data_file = os.path.join(self.person_data_path, 'DB.DAT')

        # 얼굴 정보 인식 모델 전체 초기화
        self.FD_Net, self.FV_Net, self.GENDER_Net, self.AGE_Net, self.Landmark_Net, self.HeadPose_Net, self.Glasses_Net = EFI.ETRI_Initialization(self.model_path)
        # 행동 정보 인식 모델 초기화
        self.EAR_Net = EAR.EAR_Initialization(self.model_path)

        self.list_PI = self.load_person_data()
        self.nFrameCNT = 0
        self.nTopNAction = 1
        self.bReady = True

        self.bridge = CvBridge()

        self.pub_recog = rospy.Publisher('recognitionResult', String, queue_size=1)
        rospy.Subscriber('Color_Image', Image, callback=self.image_cb)

        rospy.loginfo("[%s] initialized.", rospy.get_name())


    def load_person_data(self):
        if os.path.exists(self.person_data_file):
            list_PI = PI.readPersonalInformation()
            PI.autoRemoveRarePerson(list_PI)
            rospy.loginfo("%d person load Done." % len(list_PI))
        else:
            list_PI = []
        return list_PI


    def image_cb(self, msg):
        diff = rospy.Time.now() - msg.header.stamp        
        if diff.secs != 0 or abs(diff.nsecs) > 100000000:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img_show = copy.deepcopy(frame)
        except CvBridgeError as e:
            rospy.logerr(e)

        # 얼굴 검출 정보 / 행동인식 정보 메모리 초기화
        list_ETRIFace = []
        nMainFaceIndex = -1
        bUpdatePI = True
        nAction = -1
        action_id = -1
        nTendency = -1

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
                nAction = EAR.EAR_BodyAction_Estimation(self.EAR_Net, convertedImg)
                rospy.logdebug("Action : %s" % EAR.sAction[nAction])
                
                # 행동 정보 업데이트
                EAR.updateAction(nAction)

                # 출력을 위해 Top N 행동 종류와 확률을 string으로 받기
                actions, sActionResult = EAR.getTopNAction(self.nTopNAction, convertedImg)
                action_id = actions[0][1] # Rank 1 Action ID
                rospy.logdebug("SAction %s and %s" % (actions, sActionResult))
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

        # Personal Information Update
        #######################
        # 얼구 검출 및 중앙 얼굴 찾기
        EFI.ETRI_Face_Detection(self.FD_Net, frame, list_ETRIFace)
        nMainFaceIndex = EFI.getCenterorBiggestFaceIndex(list_ETRIFace, nCenterFaceX, nCenterFaceY)
        if nMainFaceIndex >= 0:
            rospy.logdebug('nMainFaceIndex = %d' % nMainFaceIndex)
            rospy.logdebug('face roi = %s' % list_ETRIFace[nMainFaceIndex].rt)
            # check interested
            bInterested = EFI.ETRI_Get_Interested(self.HeadPose_Net, frame, list_ETRIFace, nMainFaceIndex)
            # 출력을 위한 정보
            sInterested = "Interested : %s" % bInterested
            cv2.putText(img_show, sInterested, (15, 20), 0, 0.7, (0, 0, 200), 2)
            rospy.logdebug('Interested: %s' % bInterested)

            # Personal Information
            # update는 20frame 마다 실행 (20FPS 기준. 약 1초간격으로 업데이트)
            if self.nFrameCNT % 20 == 0 and bInterested:
                # 중앙에 있는 인물이 관심이 있는 상태에만 업데이트 진행
                # 개인 특징 추출하여 개인정보 관리
                EFI.ETRI_Landmark_Detection(self.Landmark_Net, frame, list_ETRIFace, nMainFaceIndex)

                age = EFI.ETRI_Age_Estimation(self.AGE_Net, frame, list_ETRIFace, nMainFaceIndex)
                rospy.logdebug('Age: %s' % age)

                gender = EFI.ETRI_Gender_Classification(self.GENDER_Net, frame, list_ETRIFace, nMainFaceIndex)
                rospy.logdebug('Gender: %s' % gender)

                glasses = EFI.ETRI_Glasses_Classification(self.AGE_Net, frame, list_ETRIFace, nMainFaceIndex)
                rospy.logdebug('Glasses: %s' % glasses)

                face_feature = EFI.ETRI_Get_Face_Feature(self.FV_Net, frame, list_ETRIFace, nMainFaceIndex)
                if len(face_feature) == 1:
                    return
                bUpdatePI = True
                
                pID = -1 # Presumed Person ID
                # 등록된 인원이 없을 경우 신규등록. ID는 현재 시간으로 임의배정
                if len(self.list_PI) == 0:
                    info = PI.PersonalInformation()
                    info.bUpdateAutoFeature(face_feature)
                    time_ID = "%08d" % int(time.time())
                    info.sID = time_ID
                    self.list_PI.append(info)
                # 등록된 인원이 있을 경우 신원인식 수행. 기존 등록 인원일 경우 업데이트, 새로운 인물이면 추가
                else:
                    bUpdate = False
                    for ii in range(len(self.list_PI)):
                        if self.list_PI[ii].bUpdateAutoFeature(face_feature):
                            rospy.logdebug("Existing Person! : {nIndex}({nTotal})".format(nIndex=ii, nTotal = len(self.list_PI)))
                            pID = ii
                            bUpdate = True
                            break

                    if not bUpdate:
                        info = PI.PersonalInformation()
                        info.bUpdateAutoFeature(face_feature)
                        time_ID = "%08d" % int(time.time())
                        info.sID = time_ID
                        self.list_PI.append(info)
                        pID = len(self.list_PI)-1

                # 개인 정보에 행동 및 태도 정보 업데이트.
                if nAction != -1:
                    self.list_PI[pID].bUpdateHabit(nAction)
                if nTendency != -1:
                    self.list_PI[pID].bUpdateTendency(nTendency)

                # PI.showPersonalInformation(list_PI[nMainFaceIndex])
            rospy.logdebug('[%s] number of faces: %d %d', rospy.get_name(), nMainFaceIndex, len(list_ETRIFace))
            self.publish_results(list_ETRIFace[nMainFaceIndex], action_id, bInterested)

        # 인식 결과 출력
        cv2.imshow("TT", img_show)
        nKey = cv2.waitKey(1)

        self.nFrameCNT = self.nFrameCNT + 1

    def shutdown_handler(self):
        PI.writePersonalInformation(self.list_PI)

    def publish_results(self, face_info, action_id, interested):
        json_data = {'encoding':'UTF-8', 'header': {'timestamp':0, 'source':'ETRI', 'target':['UOA', 'UOS'], 'content':'human_recognition'}}

        recog = {}
        recog['id'] = 0
        recog['age'] = face_info.fAge
        recog['name'] = ''
        if face_info.fGender == 0: # 남성
            recog['gender'] = 1
        elif face_info.fGender == 1: # 여성
            recog['gender'] = 0
        if face_info.fGlasses == 0:
            recog['glasses'] = True
        elif face_info.fGlasses == 1:
            recog['glasses'] = False
        recog['face_roi'] = {'x1':face_info.rt[0], 'y1':face_info.rt[1], 'x2':face_info.rt[2], 'y2':face_info.rt[3]}
        recog['headpose'] = {'roll':face_info.fYaw, 'pitch':face_info.fPitch, 'yaw':face_info.fYaw}
        recog['social_action'] = action_id

        if interested is True:
            recog['gaze'] = 1
        elif interested is False:
            recog['gaze'] = 0
            
        json_data['human_recognition'] = [recog]
        
        msg = json.dumps(json_data)
        rospy.logdebug('[%s] publishing: %s', rospy.get_name(), msg)
        self.pub_recog.publish(msg)


if __name__ == '__main__':
    rospy.init_node('etri_recognition_node', anonymous=False)

    m = ETRISocialBehaviorRecognitionNode()

    rospy.on_shutdown(m.shutdown_handler)

    rospy.spin()
