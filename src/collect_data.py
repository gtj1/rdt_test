import time
import os
import cv2
import json
import numpy as np
from dataclasses import dataclass
from typing import TypedDict, Literal, TypeAlias, Iterator, overload
from multiprocessing import Process, Queue

import pyrealsense2 as rs
from jodellSdk.jodellSdkDemo import RgClawControl
from robotcontrol import Auboi5Robot, RobotErrorType, logger_init

with open('src/robot_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

Pose: TypeAlias = tuple[np.ndarray, np.ndarray]  # xyz in m, rpy in rad


class ArmState(TypedDict):
    arm_name: str
    joint_position: list[float]
    end_effector_pose: Pose


class GripperState(TypedDict):
    position: float
    speed: float
    torque: float


class RobotCommand(TypedDict):
    command_type: Literal['state', 'move', 'shutdown']
    # The following fields are only used when command_type is not 'shutdown'
    arm_state: ArmState | None
    gripper_state: GripperState | None


class RobotRecord(TypedDict):
    state: RobotCommand
    action: RobotCommand


class CameraFrame(TypedDict):
    rgbd_depth: np.ndarray
    rgbd_image: np.ndarray
    usb_image: np.ndarray


RecordFrame: TypeAlias = tuple[RobotCommand, CameraFrame]


class RobotController(Process):
    arm: Auboi5Robot
    arm_ip: str
    arm_port: int
    arm_name: str

    gripper: RgClawControl
    gripper_port: str
    gripper_baud_rate: int
    gripper_slave_id: int
    gripper_max_position: float

    command_queue: Queue
    record_queue: Queue
    running: bool

    last_command: RobotCommand

    def __init__(
        self,
        command_queue: Queue,
        record_queue: Queue
    ):
        super().__init__()
        self.command_queue = command_queue
        self.record_queue = record_queue
        self.last_command = None

    def connect_to_arm(self):
        self.arm_ip = config['arm']['ip']
        self.arm_port = config['arm']['port']
        self.arm_name = config['arm']['name']

        self.arm = Auboi5Robot()
        self.arm.create_context()

        result = self.arm.connect(self.arm_ip, self.arm_port)
        if result != RobotErrorType.RobotError_SUCC:
            raise RuntimeError(f'Failed to connect to arm {self.arm_name}')

        # 1 for real arm, 0 for simulation
        self.arm.set_work_mode(1)
        self.arm.project_startup()
        self.arm.set_collision_class(config['arm']['collision_class'])
        self.arm.enable_robot_event()
        self.arm.init_profile()

    def set_arm_parameters(self):
        self.arm.set_joint_maxacc(
            config['arm']['max_joint_angular_acceleration'])
        self.arm.set_joint_maxvelc(config['arm']['max_joint_angular_velocity'])
        self.arm.set_end_max_line_acc(
            config['arm']['max_end_effector_acceleration'])
        self.arm.set_end_max_line_velc(
            config['arm']['max_end_effector_velocity'])

    def connect_to_gripper(self):
        self.gripper = RgClawControl()
        self.gripper_port = config['gripper']['port']
        self.gripper_baudrate = config['gripper']['baudrate']
        self.gripper_slave_id = config['gripper']['slave_id']
        self.gripper_max_position = config['gripper']['max_position']

        result = self.gripper.serialOperation(
            com=self.gripper_port,
            baudRate=self.gripper_baudrate,
            state=True  # open
        )
        if result != 1:
            raise RuntimeError(f'Failed to connect to gripper, info: {result}')

        result = self.gripper.enableClamp(self.gripper_slave_id, True)
        if result != 1:
            raise RuntimeError(f'Failed to enable gripper, info: {result}')
        else:
            print('Gripper enabled')

    def connect(self):
        self.connect_to_arm()
        self.set_arm_parameters()
        self.connect_to_gripper()

        self.running = True

    def arm_move_to(self, end_effector_pose: Pose) -> bool:
        xyz, rpy = end_effector_pose
        try:
            self.arm.move_to_target_in_cartesian(xyz, rpy)
            return True
        except Exception as e:
            print(f'Failed to move arm to {end_effector_pose}: {e}')
            return False

    def gripper_move_to(self, gripper_state: GripperState) -> bool:
        position = gripper_state['position']
        speed = gripper_state['speed']
        torque = gripper_state['torque']
        if position < 0 or position > self.gripper_max_position:
            print(f'Invalid gripper position: {position}')
            return False
        try:
            self.gripper.runWithParam(
                slaveId=self.gripper_slave_id,
                pos=position,
                speed=speed,
                torque=torque
            )
            return True
        except Exception as e:
            print(f'Failed to move gripper to {gripper_state}: {e}')
            return False

    def execute(self, command: RobotCommand) -> bool:
        command_type = command['command_type']
        if command_type == 'move':
            end_effector_pose = command['arm_state']['end_effector_pose']
            gripper_state = command['gripper_state']
            if end_effector_pose is None or gripper_state is None:
                raise ValueError(
                    'Pose and gripper state must be provided for move command')
            self.last_command = command
            return self.arm_move_to(end_effector_pose) and self.gripper_move_to(gripper_state)
        elif command_type == 'shutdown':
            self.running = False
            return True
        else:
            raise ValueError(f'Unknown command type: {command_type}')

    def release(self):
        if self.arm.connected:
            self.arm.disconnect()
            self.arm.uninitialize()
        print(f'{self.arm_name} controller shutdown')

    def check_command_queue(self) -> bool:
        if self.command_queue.empty():
            return
        command: RobotCommand = self.command_queue.get()
        self.execute(command)

    def get_arm_state(self) -> ArmState | None:
        current_state: dict = self.arm.get_current_waypoint()
        if current_state is None:
            return None
        joint_position = current_state['joint']
        xyz = np.array(current_state['pos'])
        rpy = np.array(self.arm.quaternion_to_rpy(current_state['ori']))
        end_effector_pose = (xyz, rpy)
        return {
            'arm_name': self.arm_name,
            'joint_position': joint_position,
            'end_effector_pose': end_effector_pose
        }

    def get_gripper_state(self) -> GripperState | None:
        gripper_position = self.gripper.getClampCurrentLocation(
            self.gripper_slave_id)
        if isinstance(gripper_position, list) and len(gripper_position) == 1:
            gripper_position = gripper_position[0]
        else:
            print(f'Invalid gripper position: {gripper_position}')
            return None
        gripper_speed = self.gripper.getClampCurrentSpeed(
            self.gripper_slave_id)
        gripper_torque = self.gripper.getClampCurrentTorque(
            self.gripper_slave_id)

        return {
            'position': gripper_position,
            'speed': gripper_speed,
            'torque': gripper_torque
        }

    def record_data(self) -> bool:
        if not self.running:
            return False

        arm_state = self.get_arm_state()
        gripper_state = self.get_gripper_state()

        if arm_state is None or gripper_state is None:
            return False

        robot_record: RobotRecord = {}
        robot_record['state'] = {
            'command_type': 'state',
            'arm_state': arm_state,
            'gripper_state': gripper_state
        }
        if self.last_command is not None:
            robot_record['action'] = self.last_command
        else:
            robot_record['action'] = {
                'command_type': 'command',
                'arm_state': arm_state,
                'gripper_state': gripper_state
            }

        self.record_queue.put({
            'command_type': 'state',
            'arm_state': arm_state,
            'gripper_state': gripper_state
        })

        return True

    def loop(self):
        self.check_command_queue()
        self.record_data()

    def run(self):
        try:
            self.connect()
            print(f'{self.arm_name} connected')
            while self.running:
                self.loop()
                time.sleep(0.1)
        except Exception as e:
            print(f'Exception in {self.arm_name} thread: {e}')
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
        self.rgbd_camera = rs.pipeline()
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
        self.rgbd_camera.start(rs_config)

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

        print('Camera initialized')

    def shot(self) -> CameraFrame:
        frames = self.rgbd_camera.wait_for_frames()
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
        self.rgbd_camera.stop()
        self.usb_camera.release()


class DataCollector(Process):
    save_path: str
    index: int = 0
    record_queue: Queue
    camera_collector: CameraCollector

    def __init__(
        self,
        record_queue: Queue,
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
        robot_record: RobotRecord,
        camera_frame: CameraFrame,
    ):
        assert robot_record['command_type'] == 'state'

        np.save(
            os.path.join(self.save_path, f'{self.index}.npy'),
            {
                'robot_record': robot_record,
                'camera_frame': camera_frame,
            },
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
                robot_record = None

                while not self.record_queue.empty():
                    latest: RobotRecord = self.record_queue.get()
                    assert latest['state']['command_type'] == 'state' and \
                        latest['action']['command_type'] == 'command'
                    robot_record = latest
                camera_frame = self.camera_collector.shot()
                if robot_record:
                    self.save(robot_record, camera_frame)

                elapsed_time = time.time() - start_time
                if elapsed_time < dt:
                    time.sleep(dt - elapsed_time)

        except KeyboardInterrupt:
            pass

        finally:
            self.camera_collector.release()


class ReplayFeeder(Process):
    dataset_path: str
    npy_files: list[str]
    command_queue: Queue

    def __init__(self, command_queue: Queue, dataset_path: str):
        super().__init__()
        self.command_queue = command_queue
        self.dataset_path = dataset_path
        self.npy_files = []

        for file in os.listdir(dataset_path):
            if file.endswith('.npy'):
                self.npy_files.append(file)
        self.npy_files.sort(key=lambda x: int(x.split('.')[0]))

    def __iter__(self) -> Iterator[tuple[RobotCommand, CameraFrame]]:
        for file in self.npy_files:
            file_path = os.path.join(self.dataset_path, file)
            load_result = np.load(file_path, allow_pickle=True).item()
            if load_result is None:
                print(f'Failed to load {file_path}')
                continue

            robot_state, camera_frame = load_result
            yield (robot_state, camera_frame)

    @overload
    def run(self):
        try:
            start_time = time.time()
            for robot_state, camera_frame in self:
                robot_state['command_type'] = 'move'
                self.command_queue.put(robot_state)
                elapsed_time = time.time() - start_time
                time.sleep(1 / config['camera']['rgbd']['frame_rate'])
                start_time = time.time()
        except KeyboardInterrupt:
            pass

# Read the robot state from the record queue and generate a new command, see RobotCommand for details
def model(robot_state: RobotCommand) -> RobotCommand: ...


if __name__ == '__main__':
    logger_init()
    replay_path = config['path']['replay']
    save_path = config['path']['save']

    command_queue: Queue = Queue()
    record_queue: Queue = Queue()

    # replay_feeder = ReplayFeeder(command_queue, replay_path)
    # data_collector = DataCollector(record_queue, save_path)

    try:
        robot_controller = RobotController(command_queue, record_queue)
    except Exception as e:
        print(f'Failed to initialize arm controller: {e}')
        exit(1)

    # replay_feeder.start()
    # data_collector.start()
    # robot_controller.start()
    
    # try:
    #     replay_feeder.join()
    #     print('Replay feeder finished')
        
    #     data_collector.join()
    #     robot_controller.join()
    # except KeyboardInterrupt:
    #     print("Exit")
    robot_controller.run()
    try:
        while True:
            if not record_queue.empty():
                robot_record: RobotRecord = record_queue.get()
                command_queue.put(model(robot_record['state']))
    except KeyboardInterrupt:
        print("Exit")
        command_queue.put({'command_type': 'shutdown'})
        robot_controller.join()
        