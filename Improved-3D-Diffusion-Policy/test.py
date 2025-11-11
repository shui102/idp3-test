'''
Descripttion: 
Author: tauceti0207
version: 
Date: 2025-11-02 15:55:49
LastEditors: tauceti0207
LastEditTime: 2025-11-03 13:41:50
'''

import sys
import os


from realman65.my_robot.realman_65_interface import Realman65Interface
import time

from termcolor import cprint




def test_gripper():
    interface = Realman65Interface(auto_setup=False)
    interface.set_up()
    # interface.set_gripper(arm_name="left_arm", value=1)
    print(interface.get_gripper_state()['left_arm'])
    # time.sleep(3)
    # interface.set_gripper(arm_name="left_arm", value=0)
    print(interface.get_gripper_state()['left_arm'])


def test_control_interface():
    interface = Realman65Interface(auto_setup=False)
    interface.set_up()
    joint_angles = interface.get_joint_angles()
    print(joint_angles)
    arm_pose = interface.get_end_effector_pose()
    print(arm_pose)
    angle = [-173.5070037841797, -82.68800354003906, -14.062999725341797,
             12.692999839782715, 7.381999969482422, -130]
    pose = [-0.606651, -0.066951, 0.25, -1.55, 1.21, 1.674]
    interface.set_end_effector_pose('left_arm', pose)
    interface.set_joint_angles('left_arm', angle)


def test_get_joint_angle():
    interface = Realman65Interface(auto_setup=False)
    interface.set_up()
    print(interface.get_joint_angles())
    joint_angles = interface.get_joint_angles()['left_arm']
    print(joint_angles)
    
    # arm_pose = interface.get_end_effector_pose()
    # print(arm_pose)
    
def test_get_ee_pose():
    interface = Realman65Interface(auto_setup=False)
    interface.set_up()
    
    # 1. 获取末端位姿
    ee_pose_dict = interface.get_end_effector_pose()
    
    # 2. 打印整体返回值（字典）的类型和内容
    cprint(f"\n=== 整体返回值信息 ===", color="blue")
    cprint(f"返回值类型：{type(ee_pose_dict)}", color="red")
    cprint(f"返回值内容：{ee_pose_dict}", color="red")

    # 3. 安全获取left_arm对应的位姿（避免key不存在或值为None）
    left_arm_pose = ee_pose_dict['left_arm']
    if left_arm_pose is not None:
        cprint(f"\n=== left_arm 位姿信息 ===", color="blue")
        cprint(f"left_arm 位姿类型：{type(left_arm_pose)}", color="red")
        cprint(f"left_arm 位姿内容：{left_arm_pose}", color="red")
        
        # 4. 打印每个元素的类型（确认是否为float）
        cprint(f"每个元素的类型：", color="blue")
        for idx, value in enumerate(left_arm_pose):
            cprint(f"  索引{idx}（{['x', 'y', 'z', 'roll', 'pitch', 'yaw'][idx]}）：值={value}，类型={type(value)}", color="red")
    else:
        cprint(f"\n警告：未获取到left_arm的末端位姿（值为None）", color="yellow")
test_get_ee_pose()
# test_control_interface()
# test_get_joint_angle()
