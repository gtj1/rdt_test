
#! /usr/bin/env python
# coding=utf-8
import time
from robotcontrol import Auboi5Robot, RobotErrorType, logger_init
from multiprocessing import Process,Queue
import cv2
import pyrealsense2 as rs
import numpy as np
import os
from jodellSdk.jodellSdkDemo import RgClawControl
import pygame



class RobotController(Process):
    def __init__(self, ip, port, name, command_queue, status_queue):
        super().__init__()
        self.ip = ip
        self.port = port
        self.robot_name = name
        self.command_queue = command_queue  # 接收指令的队列
        self.status_queue =  status_queue    # 反馈状态的队列
        self.running = True
        self.robot = None

    def _connect_robot(self):
        """连接机械臂并初始化参数"""
        self.robot = Auboi5Robot()

        #创建上下文句柄
        self.robot.create_context()

        # 连接机械臂
        result = self.robot.connect(self.ip, self.port)
        if result != RobotErrorType.RobotError_SUCC:
            raise ConnectionError(f"{self.robot_name} connection failed!")
        
        # 机械臂上电和初始化
        # 机械臂模式（真实：1      仿真：0）
        self.robot.set_work_mode(1)
        self.robot.project_startup()

        #设置机械臂碰撞等级,范围（0-10）
        self.robot.set_collision_class(6)  # 示例碰撞等级设置

        #使能机械臂回调事件
        self.robot.enable_robot_event()

        #初始化机械臂控制全局属性
        self.robot.init_profile()
        
        # 设置速度参数（rad/s）（根据型号不同调整）
        self.robot.set_joint_maxacc((0.8/2.5, 0.8/2.5, 0.8/2.5, 0.8/2.5, 0.8/2.5, 0.8/2.5))
        self.robot.set_joint_maxvelc((0.8, 0.8, 0.8, 0.8, 0.8, 0.8))

        #设置末端最大线速度和加速度（m/s）
        self.robot.set_end_max_line_acc(0.4)
        self.robot.set_end_max_line_velc(0.4)

        #设置机械臂提前到位时间（s）和距离（m）（在复现轨迹时合理调整可以优化机械臂卡顿现象）
        self.robot.set_arrival_ahead_time(0.06)
        self.robot.set_arrival_ahead_distance(0.03) 
        # self.robot.remove_all_waypoint()

    def _move_to_cartesian(self, pos, rpy_xyz):
        """移动到笛卡尔坐标系位置"""
        try:
            # 移动到指定机械臂末端位置（m）与姿态（deg）
            self.robot.move_to_target_in_cartesian(pos, rpy_xyz)
            return True
        except Exception as e:
            print(f"{self.robot_name} movement error: {str(e)}")
            return False
            

    def run(self):
        """线程主循环"""
        try:
            self._connect_robot()
            print(f"{self.robot_name} connected successfully!")
            
            while self.running:
                # 处理指令队列
                if not self.command_queue.empty():
                    cmd = self.command_queue.get()
                    if cmd['type'] == 'move':
                        self._move_to_cartesian(cmd['pos'], cmd['rpy'])
                    elif cmd['type'] == 'shutdown':
                        break
                
                # 反馈当前状态
                current_pos = self.robot.get_current_waypoint()
                if current_pos:
                    # 将四元数转换为欧拉角（RPY）
                    rpy=self.robot.quaternion_to_rpy(current_pos['ori'])

                    #弧度->度
                    rpy_deg=list(tuple(np.degrees(rpy)))

                    # 将位置（m）和姿态（deg）拼接为一个六维向量
                    position_and_rpy=current_pos['pos']+rpy_deg

                    #更新位置信息
                    self.status_queue.put({
                        'robot': self.robot_name,
                        'type': 'status_update',
                        'position': position_and_rpy,           
                        'joints': current_pos['joint'],
                        'gripper': 0  # 新增夹爪状态
                    })
                
                time.sleep(0.05)  # 控制状态更新频率
                
        except Exception as e:
            print(f"{self.robot_name} controller error: {str(e)}")
        finally:
            # 确保在退出时关闭机器人连接
            if self.robot.connected:
                self.robot.disconnect()
                self.robot.uninitialize()
            print(f"{self.robot_name} controller shutdown.")


class CameraCollector:
    def __init__(self):
        # RealSense配置
        self.pipeline = rs.pipeline()
        config = rs.config()

        #RealSense相机参数
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        # USB相机配置（假设设备索引为0）
        self.usb_cam = cv2.VideoCapture(0)  
        self.usb_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.usb_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

        # 预热：丢弃前100帧让相机完成对焦
        for _ in range(100):
            ret, _ = self.usb_cam.read()
            if not ret:
                raise RuntimeError("USB相机预热失败")
                
        print("相机初始化完成，已丢弃100帧完成预热")

    def get_frames(self):
        # RealSense帧
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        # 处理深度图像
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # 处理彩色图像
        color_image = np.asanyarray(color_frame.get_data())
        
        # 调整尺寸到320x256
        resized_color = cv2.resize(color_image, (320, 256))
        resized_depth = cv2.resize(depth_colormap, (320, 256))
        
        # USB相机帧
        ret, usb_frame = self.usb_cam.read()
        if not ret:
            raise RuntimeError("无法读取USB相机")
            
        return {
            'wrist_rgb': resized_color,
            'wrist_depth': resized_depth,
            'third_view': usb_frame
        }

    def release(self):
        # 释放资源
        self.pipeline.stop()
        self.usb_cam.release()

class DataCollector:
    def __init__(self, save_path):
        self.save_path = save_path
        self.index = 0
        
    def save(self, robot_status, images):
        # 保存数据
        data = {
            'joint': np.array(robot_status['joints'], dtype=np.float32),
            'pose': np.array(robot_status['position'], dtype=np.float32),
            'gripper': robot_status['gripper'],
            'wrist_rgb': images['wrist_rgb'],
            'wrist_depth': images['wrist_depth'],
            'third_view': images['third_view']
        }
        
        # 保存图像
        cv2.imwrite(f"{self.save_path}/{self.index}_wrist_rgb.jpg", images['wrist_rgb'])
        cv2.imwrite(f"{self.save_path}/{self.index}_wrist_depth.jpg", images['wrist_depth'])
        cv2.imwrite(f"{self.save_path}/{self.index}_third_view.jpg", images['third_view'])
        
        # 保存npy
        np.save(f"{self.save_path}/{self.index}_data.npy", data)
        self.index += 1



# 示例使用
if __name__ == "__main__":
    logger_init()  # 初始化日志
    cemera = CameraCollector()

    # 初始化数据收集器
    save_path = "./src/dataset_95"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_collector=DataCollector(save_path)
    
    # 初始化手柄
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("未检测到手柄！")
        exit()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    # 初始化死区
    DEADZONE = 0.25

    # 定义夹爪
    clawTool = None
    gripper_pos = -1
    
    try:
        # 初始化夹爪
        clawTool = RgClawControl()

        # 初始化夹爪串口
        port = '/dev/ttyUSB0'

        #初始化波特率
        baud_rate = 115200

        flag = clawTool.serialOperation(port, baud_rate, True)
        if flag != 1:
            print(f"夹爪初始化失败：{flag}")
        else:
            #夹爪使能（注意使能时会完全闭合）
            result = clawTool.enableClamp(9, True)
            if result != 1:
                print("夹爪使能失败")
            else:
                print("夹爪就绪！")
    except Exception as e:
        print(f"夹爪初始化异常：{e}")


    # 创建指令队列和状态队列
    command_queue_1 = Queue()
    status_queue = Queue()

    # 初始化机械臂控制器
    robot1 = RobotController(
        ip="192.168.1.41", 
        port=8899,
        name="Robot_E3",
        command_queue=command_queue_1,
        status_queue=status_queue
    )

    # 启动线程
    robot1.start()

    prev_rt_pressed = False
    prev_lt_pressed = False

    # 监控状态反馈
    try:
        while True:
            start_time=time.time()
            pygame.event.pump()  # 处理手柄事件

            # 获取手柄输入
            axis_0 = joystick.get_axis(0)  # 左摇杆X
            axis_1 = joystick.get_axis(1)  # 左摇杆Y
            axis_3 = joystick.get_axis(3)  # 右摇杆X
            axis_4 = joystick.get_axis(4)  # 右摇杆Y
            dpad = joystick.get_hat(0)     # 方向键

          # 应用死区处理
            def apply_deadzone(value):
                return 0.0 if abs(value) < DEADZONE else value
            axis_0 = apply_deadzone(axis_0)
            axis_1 = apply_deadzone(axis_1)
            axis_3 = apply_deadzone(axis_3)
            axis_4 = apply_deadzone(axis_4)

            # 计算机械臂移动增量（调整缩放因子控制速度）
            delta_x = axis_0 * 0.02
            delta_y = -axis_1 * 0.02  # 反转Y轴
            delta_z = -axis_4 * 0.02
            delta_roll = dpad[0] * 5  # 左右方向键控制roll（度）
            delta_pitch = -dpad[1] * 5 # 上下方向键控制pitch（度）
            delta_yaw = axis_3 * 5    # 右摇杆X控制yaw（度）

            # 获取最新状态
            latest_status=None
            while not status_queue.empty():
                latest_status=status_queue.get()

            if latest_status:
                # 计算输入量的绝对值总和
                input_magnitude = sum(abs(v) for v in [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw])
                
                # 当检测到有效输入时
                if input_magnitude > 0.001:
                    current_pos = latest_status['position']
                    new_pos = [
                        current_pos[0] + delta_x,
                        current_pos[1] + delta_y,
                        current_pos[2] + delta_z,
                        current_pos[3] + delta_roll,
                        current_pos[4] + delta_pitch,
                        current_pos[5] + delta_yaw
                ]
                    # 清除队列中旧的移动指令（保留其他类型指令）
                    temp_commands = []
                    while not command_queue_1.empty():
                        cmd = command_queue_1.get()
                        if cmd['type'] != 'move':
                            temp_commands.append(cmd)
                    # 重新放入非移动指令
                    for cmd in temp_commands:
                        command_queue_1.put(cmd)
                # 发送笛卡尔移动指令（注意单位转换）

                    command_queue_1.put({
                        'type': 'move',
                        'pos': new_pos[:3],
                        'rpy': new_pos[3:6]
                    })
                else:
                    # 当输入量接近零时，清空所有移动指令
                    temp_commands = []
                    while not command_queue_1.empty():
                        cmd = command_queue_1.get()
                        if cmd['type'] != 'move':
                            temp_commands.append(cmd)
                    for cmd in temp_commands:
                        command_queue_1.put(cmd)

            # 处理夹爪控制
            rt = joystick.get_axis(5)
            lt = joystick.get_axis(2)
            rt_pressed = rt > DEADZONE
            lt_pressed = lt > DEADZONE

            if clawTool is not None:
                if rt_pressed and not prev_rt_pressed:
                    clawTool.runWithParam(9, 115, 100, 30)
                elif lt_pressed and not prev_lt_pressed:
                    clawTool.runWithParam(9, 0, 100, 30)
            prev_rt_pressed, prev_lt_pressed = rt_pressed, lt_pressed


            # 获取图像
            images = cemera.get_frames()
            # 获取夹爪状态

            if latest_status:
                # 更新夹爪状态
                try:
                    if clawTool is not None:
                        #获取夹爪位置
                        pos = clawTool.getClampCurrentLocation(9)[0]
                        #归一化到0-1之间
                        latest_status['gripper'] = round(1.0 - pos/255.0, 4)
                except Exception as e:
                    print(f"夹爪状态获取失败: {e}")
                    
                #保存数据
                data_collector.save(latest_status, images)
            
            # 控制采集频率（约20hz）
            elapsed=time.time()-start_time
            if elapsed<0.05:
                time.sleep(0.05-elapsed)
            
    except KeyboardInterrupt:
        # 安全关闭
        command_queue_1.put({'type': 'shutdown'})
        robot1.join()
        cemera.release()
        if clawTool is not None:
            clawTool.enableClamp(9, False)
            clawTool.serialOperation(port,baud_rate,False)
        pygame.quit()
        print("数据采集完成，资源已释放")