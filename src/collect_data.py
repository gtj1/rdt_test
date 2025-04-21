import time
import os
import cv2
import json
import numpy as np
from dataclasses import dataclass
from typing import TypedDict, Literal, TypeAlias, Iterable, overload
from multiprocessing import Process, Queue

import pyrealsense2 as rs
from jodellSdk.jodellSdkDemo import RgClawControl
from robotcontrol import Auboi5Robot, RobotErrorType, logger_init

with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

Pose: TypeAlias = tuple[np.ndarray, np.ndarray]  # xyz in m, rpy in rad

class GripperState(TypedDict):
    position: float
    speed: float
    torque: float

class RobotCommand(TypedDict):
    robot_name: str
    command_type: Literal['record', 'move', 'shutdown']
    # The following fields are only used when command_type is not 'shutdown'
    pose: Pose | None
    joint_position: list[float] | None
    gripper_state: GripperState | None

class CameraFrame(TypedDict):
    rgbd_depth: np.ndarray
    rgbd_color: np.ndarray
    usb_image: np.ndarray

class RobotController(Process):
    ip: str
    port: int

    robot: Auboi5Robot
    robot_name: str

    gripper: RgClawControl
    gripper_port: str
    gripper_baud_rate: int
    gripper_slave_id: int
    gripper_max_position: float

    command_queue: Queue[RobotCommand]
    record_queue: Queue[RobotCommand]
    running: bool

    def __init__(
        self,
        command_queue: Queue[RobotCommand],
        record_queue: Queue[RobotCommand]
    ):
        super().__init__()
        self.command_queue = command_queue
        self.record_queue = record_queue

    def connect(self):
        super().__init__()

        self.ip = config['sysem']['ip']
        self.port = config['sysem']['port']
        self.robot_name = config['robot']['name']

        self.robot = Auboi5Robot()
        self.robot.create_context()

        result = self.robot.connect(self.ip, self.port)
        if result != RobotErrorType.RobotError_SUCC:
            raise RuntimeError(f"Failed to connect to robot {self.robot_name}")
        
        # 1 for real robot, 0 for simulation
        self.robot.set_work_mode(1)
        self.robot.project_startup()
        self.robot.set_collision_class(config['robot']['collision_class'])
        self.robot.enable_robot_event()
        self.robot.init_profile()

        self.robot.set_joint_maxacc(config['robot']['max_joint_angular_acceleration'])
        self.robot.set_joint_maxvelc(config['robot']['max_joint_angular_velocity'])
        self.robot.set_end_max_line_acc(config['robot']['max_end_effector_acceleration'])
        self.robot.set_end_max_line_velc(config['robot']['max_end_effector_velocity'])

        self.gripper = RgClawControl()
        self.gripper_port = config['gripper']['port']
        self.gripper_baudrate = config['gripper']['baudrate']
        self.gripper_slave_id = config['gripper']['slave_id']
        self.gripper_max_position = config['gripper']['max_position']
        result = self.gripper.serialOperation(
            com = self.gripper_port,
            baudrate = self.gripper_baudrate,
            state = True # open
        )
        if result != 1:
            raise RuntimeError(f"Failed to connect to gripper, info: {result}")
        
        result = self.gripper.enableClamp(self.gripper_slave_id, True)
        if result != 1:
            raise RuntimeError(f"Failed to enable gripper, info: {result}")
        else:
            print("Gripper enabled")

        self.running = True
    
    def robot_move_to(self, pose: Pose) -> bool:
        xyz, rpy = pose
        try:
            self.robot.move_to_target_in_cartesian(xyz, rpy)
            return True
        except Exception as e:
            print(f"Failed to move robot to {pose}: {e}")
            return False
        
    def gripper_move_to(self, gripper_state: GripperState) -> bool:
        position = gripper_state['position']
        speed = gripper_state['speed']
        torque = gripper_state['torque']
        if position < 0 or position > self.gripper_max_position:
            print(f"Invalid gripper position: {position}")
            return False
        try:
            self.gripper.runWithParam(
                slaveId = self.gripper_slave_id,
                pos = position,
                speed = speed,
                torque = torque
            )
            return True
        except Exception as e:
            print(f"Failed to move gripper to {gripper_state}: {e}")
            return False
    
    def set_shutdown_command(self) -> None:
        self.command_queue.put(
            {'robot_name': self.robot_name, 'command': 'shutdown'}
        )
        
    def release(self):
        if self.robot.connected:
            self.robot.disconnect()
            self.robot.uninitialize()
        print(f"{self.robot_name} controller shutdown")
        
    def check_command_queue(self) -> bool:
        if self.command_queue.empty():
            return
        command = self.command_queue.get()
        command_type = command['command_type']
        if command_type == 'stop':
            self.running = False
            return True
        elif command_type != 'move':
            raise ValueError(f"Unknown command type: {command_type}")
        robot_result = self.robot_move_to(command['pose'])
        gripper_result = self.gripper_move_to(command['gripper_state'])
        return robot_result and gripper_result

    def record_data(self) -> bool:
        if not self.running:
            return
        current_state: dict = self.robot.get_current_waypoint()
        if current_state is None:
            return False
        joint_position = current_state['joint']
        xyz = current_state['pos']
        rpy = self.robot.quaternion_to_rpy(current_state['ori'])
        pose = (xyz, rpy)

        gripper_position = self.gripper.getClampCurrentLocation(self.gripper_slave_id)
        gripper_speed = self.gripper.getClampCurrentSpeed(self.gripper_slave_id)
        gripper_torque = self.gripper.getClampCurrentTorque(self.gripper_slave_id)
        gripper_state: GripperState = {
            'position': gripper_position,
            'speed': gripper_speed,
            'torque': gripper_torque
        }

        if isinstance(gripper_position, list) and len(gripper_position) == 1:
            gripper_position = gripper_position[0]
        else:
            gripper_position = -1 # invalid

        self.record_queue.put({
            'robot_name': self.robot_name,
            'command_type': 'record',
            'pose': pose,
            'joint_position': joint_position,
            'gripper_state': gripper_state
        })
        
    def loop(self):
        self.check_command_queue()
        self.record_data()

    def run(self):
        try:
            self.connect()
            print(f"{self.robot_name} connected")
            while self.running:
                self.loop()
                time.sleep(0.1)
        except Exception as e:
            print(f"Exception in {self.robot_name} thread: {e}")
        finally:
            self.release()

class CameraCollector:
    rgbd_camera: rs.pipeline
    rgbd_width: int
    rgbd_height: int
    rgbd_frame_rate: int

    usb_camera: cv2.VideoCapture
    usb_width: int
    usb_height: int

    resize_width: int
    resize_height: int

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.rgbd_width = config['camera']['rgbd']['width']
        self.rgbd_height = config['camera']['rgbd']['height']
        self.rgbd_frame_rate = config['camera']['rgbd']['frame_rate']
   
        rs_config = rs.config()
        rs_config.enable_stream(
            rs.stream.depth, self.rgbd_width, self.rgbd_height, rs.format.z16, self.rgbd_frame_rate
        )
        rs_config.enable_stream(
            rs.stream.color, self.rgbd_width, self.rgbd_height, rs.format.bgr8, self.rgbd_frame_rate
        )
        self.pipeline.start(rs_config)

        self.usb_camera = cv2.VideoCapture(0)
        self.usb_width = config['camera']['usb']['width']
        self.usb_height = config['camera']['usb']['height']

        self.usb_camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.usb_width)
        self.usb_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.usb_height)

        for _ in range(config['camera']['usb']['warmup_frames']):
            result, _ = self.usb_camera.read()
            if not result:
                raise RuntimeError('USB camera warmup failed')
            
        self.resize_width = config['camera']['resize']['width']
        self.resize_height = config['camera']['resize']['height']
        
        print("Camera initialized")

    def get_frame(self) -> CameraFrame:
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        rgbd_depth = cv2.resize(
            np.asarray(depth_frame.get_data()), 
            (self.resize_width, self.resize_height)
        )
        rgbd_color = cv2.resize(
            np.asarray(color_frame.get_data()), 
            (self.resize_width, self.resize_height)
        )
        
        result, usb_frame = self.usb_camera.read()
        if not result:
            raise RuntimeError('USB camera read failed')
        
        return {
            'rgbd_depth': rgbd_depth,
            'rgbd_color': rgbd_color,
            'usb_image': usb_frame
        }
    
    def release(self):
        self.pipeline.stop()
        self.usb_camera.release()

class DataCollector(Process):
    save_path: str
    index: int = 0
    record_queue: Queue[RobotCommand]
    camera_collector: CameraCollector

    def __init__(
        self, 
        record_queue: Queue[RobotCommand],
        save_path: str
    ):
        super().__init__()

        self.index = 0
        self.save_path = save_path
        self.record_queue = record_queue
        self.camera_collector = None

        os.makedirs(self.save_path, exist_ok=True)

    def save(
        self,
        robot_state: RobotCommand,
        camera_frame: CameraFrame,
    ):
        assert robot_state['command_type'] == 'record'

        np.save(
            f"{self.save_path}/{self.index}.npy",
            (robot_state, camera_frame),
            allow_pickle=True
        )
        self.index += 1

    @overload
    def run(self) -> None:
        self.camera_collector = CameraCollector()
        dt = 1 / config['camera']['rgbd']['frame_rate']
        try:
            while True:
                start_time = time.time()
                robot_state = None

                while not self.record_queue.empty():
                    latest = self.record_queue.get()
                    if latest['command_type'] == 'record':
                        robot_state = latest
                camera_frame = self.camera_collector.get_frame()
                if robot_state:
                    self.save(robot_state, camera_frame)

                elapsed_time = time.time() - start_time
                if elapsed_time < dt:
                    time.sleep(dt - elapsed_time)

        except KeyboardInterrupt:
            pass

        finally:
            self.camera_collector.release()


class DataFeeder(Process):
    dataset_path: str
    npy_files: list[str]
    command_queue: Queue[RobotCommand]

    def __init__(self, command_queue: Queue[RobotCommand], dataset_path: str):
        super().__init__()
        self.command_queue = command_queue
        self.dataset_path = dataset_path
        self.npy_files = []
        
        for file in os.listdir(dataset_path):
            if file.endswith('.npy'):
                self.npy_files.append(file)
        self.npy_files.sort(key=lambda x: int(x.split('.')[0]))
    
    def __iter__(self) -> Iterable[tuple[RobotCommand, CameraFrame]]:
        for file in self.npy_files:
            file_path = os.path.join(self.dataset_path, file)
            robot_state, camera_frame = np.load(file_path, allow_pickle=True).item()
            yield robot_state, camera_frame
    
    @overload
    def run(self):
        for robot_state, camera_frame in self:
            robot_state['command_type'] = 'move'
            self.command_queue.put(robot_state)

if __name__ == 'main':
    logger_init()
    save_path = config['save_path']

    command_queue = Queue()
    record_queue = Queue()

    camera = CameraCollector()
    data_feeder = DataFeeder(command_queue, save_path)
    data_collector = DataCollector(record_queue, save_path)

    try:
        robot_controller = RobotController(command_queue, record_queue)
    except Exception as e:
        print(f"Failed to initialize robot controller: {e}")
        exit(1)


