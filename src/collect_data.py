import time
import os
import cv2
import json
import numpy as np
import numpy.typing as npt
import pygame
# from dataclasses import dataclass
from typing import TypedDict, Literal, TypeAlias, Iterator, TypeVar
from multiprocessing import Process

# import a wrapped `Queue` so that it supports generic types like `Queue[RobotCommand]`
from .refined_queue import Queue

import pyrealsense2 as rs
from jodellSdk.jodellSdkDemo import RgClawControl
from .robotcontrol import Auboi5Robot, RobotErrorType, logger_init

with open('src/robot_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

T = TypeVar('T')

Pose: TypeAlias = tuple[
    tuple[float, float, float],  # xyz in m
    tuple[float, float, float]   # rpy in radians
]
JointPosition: TypeAlias = tuple[float, float, float, float, float, float]

class ArmState(TypedDict):
    arm_name: str
    joint_position: JointPosition
    end_effector_pose: Pose


class GripperState(TypedDict):
    position: int
    speed: int
    torque: int


class RobotCommand(TypedDict):
    command_type: Literal['state', 'move', 'shutdown']
    # The following fields are only used when command_type is not 'shutdown'
    arm_state: ArmState | None
    gripper_state: GripperState | None


class RobotRecord(TypedDict):
    state: RobotCommand
    action: RobotCommand


class CameraFrame(TypedDict):
    rgbd_depth: npt.NDArray[np.uint8]
    rgbd_image: npt.NDArray[np.uint8]
    usb_image: npt.NDArray[np.uint8]


class JoystickInputInfo(TypedDict):
    type: Literal['axis', 'button', 'hat', 'ball']
    id: int
    index: Literal[0, 1] | None
    scale: float


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

    command_queue: Queue[RobotCommand]
    record_queue: Queue[RobotRecord] | None
    running: bool

    last_command: RobotCommand | None

    def __init__(
        self,
        command_queue: Queue[RobotCommand],
        record_queue: Queue[RobotRecord] | None = None,
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
    
    def joint_to_arm_state(self, joint_position: JointPosition) -> ArmState:
        waypoint = self.arm.forward_kin(joint_position)
        assert waypoint is not None

        orientation = waypoint['ori']
        rpy = self.arm.quaternion_to_rpy(orientation)
        assert rpy is not None

        xyz = waypoint['pos']

        return ArmState(
            arm_name=self.arm_name,
            joint_position=joint_position,
            end_effector_pose=(xyz, rpy)
        )
    
    def pose_to_arm_state(self, end_effector_pose: Pose) -> ArmState:
        if self.last_command is None or\
            self.last_command['arm_state'] is None:
            current_state = self.arm.get_current_waypoint()
            assert current_state is not None
            last_joint_angle = current_state['joint']
        else:
            last_joint_angle = self.last_command['arm_state']['joint_position']

        xyz, rpy = end_effector_pose
        orientation = self.arm.rpy_to_quaternion(rpy)

        assert orientation is not None

        waypoint = self.arm.inverse_kin(
            joint_radian=last_joint_angle,
            pos=xyz,
            ori=orientation
        )

        assert waypoint is not None

        joint_position = waypoint['joint']

        return ArmState(
            arm_name=self.arm_name,
            joint_position=joint_position,
            end_effector_pose=(xyz, rpy)
        )

    def arm_move_to(self, joint_position: JointPosition) -> bool:
        try:
            self.arm.move_joint(joint_position)
            return True
        except Exception as e:
            print(f'Failed to move arm to joint angles {joint_position}: {e}')
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
                self.gripper_slave_id,
                position,
                speed,
                torque
            )
            return True
        except Exception as e:
            print(f'Failed to move gripper to {gripper_state}: {e}')
            return False

    def execute(self, command: RobotCommand) -> bool:
        command_type = command['command_type']
        if command_type == 'move':
            if command['arm_state'] is None or command['gripper_state'] is None:
                raise ValueError(
                    'Pose and gripper state must be provided for move command')
            joint_position = command['arm_state']['joint_position']
            end_effector_pose = command['arm_state']['end_effector_pose']
            gripper_state = command['gripper_state']
            # ensure that both joint_position and end_effector_pose can be used
            if np.nan not in end_effector_pose[0]:
                command['arm_state'] = self.pose_to_arm_state(end_effector_pose)
            elif np.nan not in joint_position:
                arm_state = self.joint_to_arm_state(joint_position)
                command['arm_state'] = arm_state
            else:
                raise ValueError(
                    'Either joint_position or end_effector_pose must be provided for move command')

            self.last_command = command
            return self.arm_move_to(command['arm_state']['joint_position']) and self.gripper_move_to(gripper_state)
        elif command_type == 'shutdown':
            self.running = False
            return True
        elif command_type == 'state':
            return self.record_data()
        else:
            raise ValueError(f'Unknown command type: {command_type}')

    def release(self):
        if self.arm.connected:
            self.arm.disconnect()
            self.arm.uninitialize()
        self.gripper.enableClamp(self.gripper_slave_id, False)
        self.gripper.serialOperation(
            com=self.gripper_port,
            baudRate=self.gripper_baudrate,
            state=False  # close
        )
        print(f'{self.arm_name} controller shutdown')

    def check_command_queue(self) -> None:
        if self.command_queue.empty():
            return
        command: RobotCommand = self.command_queue.get()
        self.execute(command)

    def get_arm_state(self) -> ArmState | None:
        current_state = self.arm.get_current_waypoint()
        if current_state is None:
            return None
        joint_position = current_state['joint']
        xyz = current_state['pos']
        rpy = self.arm.quaternion_to_rpy(current_state['ori'])
        assert rpy is not None

        end_effector_pose = (xyz, rpy)
        return ArmState(
            arm_name=self.arm_name,
            joint_position=joint_position,
            end_effector_pose=end_effector_pose
        )

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
        if isinstance(gripper_speed, list) and len(gripper_speed) == 1:
            gripper_speed = gripper_speed[0]
        else:
            print(f'Invalid gripper speed: {gripper_speed}')
            return None
        gripper_torque = self.gripper.getClampCurrentTorque(
            self.gripper_slave_id)
        if isinstance(gripper_torque, list) and len(gripper_torque) == 1:
            gripper_torque = gripper_torque[0]
        else:
            print(f'Invalid gripper torque: {gripper_torque}')
            return None

        return {
            'position': gripper_position,
            'speed': gripper_speed,
            'torque': gripper_torque
        }

    def record_data(self) -> bool:
        if not self.running or self.record_queue is None:
            return False

        arm_state = self.get_arm_state()
        gripper_state = self.get_gripper_state()

        if arm_state is None or gripper_state is None:
            return False

        robot_record_state = RobotCommand(
            command_type='state',
            arm_state=arm_state,
            gripper_state=gripper_state
        )
        if self.last_command is not None:
            robot_record_action = self.last_command
        else:
            robot_record_action = RobotCommand(
                command_type='move',
                arm_state=arm_state,
                gripper_state=gripper_state
            )
        
        robot_record = RobotRecord(
            state=robot_record_state,
            action=robot_record_action
        )

        self.record_queue.put(robot_record)

        return True

    def loop(self):
        self.check_command_queue()
        # self.record_data()

    def run(self):
        try:
            self.connect()
            print(f'{self.arm_name} connected')
            while self.running:
                self.loop()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f'Exception in {self.arm_name} thread: {e}')
        finally:
            self.release()
            print("Robot controller process exit")


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

        rgbd_depth = np.asarray(cv2.resize(
            np.asarray(depth_frame.get_data(), dtype=np.uint8),
            (self.resize_width, self.resize_height)
        ), dtype=np.uint8)
        rgbd_color = np.asarray(cv2.resize(
            np.asarray(color_frame.get_data()),
            (self.resize_width, self.resize_height)
        ), dtype=np.uint8)

        result, usb_frame = self.usb_camera.read()
        if not result:
            raise RuntimeError('USB camera read failed')
        usb_frame = np.asarray(usb_frame, dtype=np.uint8)

        return CameraFrame(
            rgbd_depth=rgbd_depth,
            rgbd_image=rgbd_color,
            usb_image=usb_frame
        )

    def release(self):
        self.rgbd_camera.stop()
        self.usb_camera.release()


class DataCollector(Process):
    save_path: str
    index: int = 0
    record_queue: Queue[RobotRecord]
    camera_collector: CameraCollector | None

    def __init__(
        self,
        record_queue: Queue[RobotRecord],
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
        assert robot_record['state']['command_type'] == 'state'

        np.save(
            os.path.join(self.save_path, f'{self.index}.npy'),
            np.array({
                'robot_record': robot_record,
                'camera_frame': camera_frame,
            }),
            allow_pickle=True
        )
        self.index += 1

    
    def run(self) -> None:
        self.camera_collector = CameraCollector()
        dt = 1 / config['camera']['rgbd']['frame_rate']
        try:
            while True:
                start_time = time.time()
                robot_record = None

                # Get latest record from the queue
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
        except Exception:
            pass

        finally:
            self.camera_collector.release()


class ReplayFeeder(Process):
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

    def __iter__(self) -> Iterator[tuple[RobotCommand, CameraFrame]]:
        for file in self.npy_files:
            file_path = os.path.join(self.dataset_path, file)
            load_result = np.load(file_path, allow_pickle=True).item()
            if load_result is None:
                print(f'Failed to load {file_path}')
                continue

            robot_state, camera_frame = load_result
            yield (robot_state, camera_frame)

    def run(self):
        try:
            start_time = time.time()
            for robot_state, _ in self:
                robot_state['command_type'] = 'move'
                self.command_queue.put(robot_state)
                elapsed_time = time.time() - start_time
                time.sleep(1 / config['camera']['rgbd']['frame_rate'] - elapsed_time)
                start_time = time.time()
        except KeyboardInterrupt:
            pass
        

class JoystickFeeder(Process):
    last_command_time: float
    joystick: pygame.joystick.JoystickType
    state_queue: Queue[RobotCommand]
    command_queue: Queue[RobotCommand]
    deadzone: float
    mapping: dict[str, JoystickInputInfo]
    polling_interval: float
    
    def __init__(self, state_queue: Queue[RobotCommand], command_queue: Queue[RobotCommand]):
        super().__init__()
        self.state_queue = state_queue
        self.command_queue = command_queue
        
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError('No joystick found')
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.deadzone = config['joystick']['deadzone']
        self.mapping = config['joystick']['mapping']
        self.last_command_time = time.time()
        self.polling_interval = config['joystick']['polling_interval']

    def apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        return value
    
    def get_input_state(self, input_name: str) -> float:
        """
        Returns the current value of the specified input based on the joystick mapping.
        """
        input_info = self.mapping[input_name]
        if input_info['type'] == 'axis':
            value = self.joystick.get_axis(input_info['id'])
        elif input_info['type'] == 'button':
            value = self.joystick.get_button(input_info['id'])
        elif input_info['type'] == 'hat':
            hat_values = self.joystick.get_hat(input_info['id'])
            assert isinstance(hat_values, tuple) and input_info['index'] is not None
            value = hat_values[input_info['index']]
        elif input_info['type'] == 'ball':
            raise NotImplementedError('Ball input not supported')
        else:
            raise ValueError(f'Unknown input type: {input_info["type"]}')
        return self.apply_deadzone(value) * input_info['scale']
    
    def get_robot_state(self) -> RobotCommand | None:
        if not self.state_queue.empty():
            print("[WARNING] state queue not empty")
        self.command_queue.put(
            RobotCommand(command_type='state', arm_state=None, gripper_state=None
        ))
        robot_state = self.state_queue.get()
        while not self.state_queue.empty():
            latest_robot_state: RobotCommand = self.state_queue.get()
            if latest_robot_state['command_type'] == 'state':
                robot_state = latest_robot_state
            # else:
            #     raise ValueError(
            #         f'Invalid command type in state queue: {latest_robot_state["command_type"]}'
            #     )
        return robot_state

    def process_state(self) -> RobotCommand | None:
        input_state: dict[str, float] = {}
        pygame.event.pump()
        for input_name in self.mapping.keys():
            input_state[input_name] = self.get_input_state(input_name)
        gripper_open = input_state.pop('gripper open')
        gripper_close = input_state.pop('gripper close')
        gripper_movement = self.apply_deadzone(gripper_open - gripper_close) * 0.05
        input_state['gripper'] = gripper_movement
        print(gripper_open, gripper_close)

        if all(value == 0 for value in input_state.values()):
            self.last_command_time = time.time()
            return None
        robot_state = self.get_robot_state()
        if robot_state is None:
            self.last_command_time = time.time()
            return None
        # robot_state['command_type'] = 'move'
        # return robot_state
        if robot_state['command_type'] != 'state':
            raise ValueError(
                f'Invalid command type in state queue: {robot_state["command_type"]}'
            )
        if robot_state['arm_state'] is None or robot_state['gripper_state'] is None:
            raise ValueError(
                'Arm state and gripper state must be provided in state queue'
            )
        
        arm_state = robot_state['arm_state']
        end_effector_pose = arm_state['end_effector_pose']
        xyz, rpy = end_effector_pose
        dt = time.time() - self.last_command_time
        # dt = 1
        # print(xyz, rpy, dt, input_state)
        xyz += np.array([
            input_state['x'], input_state['y'], input_state['z']
        ]) * dt
        rpy += np.array([
            input_state['roll'], input_state['pitch'], input_state['yaw']
        ]) * dt
        xyz = tuple(xyz.tolist())
        rpy = tuple(rpy.tolist())
        new_arm_state = ArmState(
            arm_name=arm_state['arm_name'],
            joint_position=arm_state['joint_position'],
            end_effector_pose=(xyz, rpy)
        )
        original_position = robot_state['gripper_state']['position']
        if abs(gripper_movement) > 1:
            max_position = config['gripper']['max_position']
            new_position = int(np.clip(original_position + gripper_movement * max_position, 0, max_position))
            print("new position:", new_position)
            speed = 1
            torque = int(max_position * abs(gripper_movement))
        else:
            new_position = original_position
            speed = 0
            torque = 0
        
        new_gripper_state = GripperState(
            position=new_position,
            speed=speed,
            torque=torque
        )
        self.last_command_time = time.time()
        # print(time.time(), robot_command['arm_state'])
        return RobotCommand(
            command_type='move',
            arm_state=new_arm_state,
            gripper_state=new_gripper_state
        )
        
    def run(self) -> None:
        try:
            while True:
                robot_command = self.process_state()
                if robot_command is not None:
                    self.command_queue.put(robot_command)
                time.sleep(self.polling_interval)
        except KeyboardInterrupt:
            pygame.quit()
            self.command_queue.put(
                RobotCommand(command_type="shutdown", arm_state=None, gripper_state=None)
            )
            print('Joystick feeder process exit')
                
    

# Read the robot state from the record queue and generate a new command, see RobotCommand for details
def model(robot_state: RobotRecord) -> RobotCommand: ...


# if __name__ == '__main__':
#     logger_init()
#     replay_path = config['path']['replay']
#     save_path = config['path']['save']

#     command_queue: Queue[RobotCommand] = Queue()
#     record_queue: Queue[RobotRecord] = Queue()

#     # replay_feeder = ReplayFeeder(command_queue, replay_path)
#     # data_collector = DataCollector(record_queue, save_path)

#     print('Start collecting data...')

#     try:
#         robot_controller = RobotController(command_queue, record_queue)
#         robot_controller.start()
#         time.sleep(1)
#     except Exception as e:
#         print(f'Failed to initialize arm controller: {e}')
#         exit(1)

#     joystick_feeder = JoystickFeeder(record_queue, command_queue)
#     joystick_feeder.start()
    
#     try:
#         while True:
#             robot_controller.join(1)
#             joystick_feeder.join(1)
#     except KeyboardInterrupt:
#         print("[INFO] Sending shutdown message")
#         command_queue.put(RobotCommand(
#             command_type='shutdown', arm_state=None, gripper_state=None
#         ))
#     finally:
#         time.sleep(1)
#         print("Main process exit")

# a simple test of the arm controller
if __name__ == '__main__':
    logger_init()
    command_queue = Queue[RobotCommand]()
    
    try:
        robot_controller = RobotController(command_queue)
        robot_controller.start()
    except Exception as e:
        print(f'Failed to initialize arm controller: {e}')
        exit(1)

    arm_state = ArmState(
        arm_name='arm',
        joint_position=(0.541678, 0.225068, -0.948709, 0.397018, -1.570800, 0.541673),
        end_effector_pose=((np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan))
    )
    command = RobotCommand(
        command_type='move', arm_state=None, gripper_state=None
    )
    command_queue.put(command)
    time.sleep(10)
    command = RobotCommand(
        command_type='shutdown', arm_state=None, gripper_state=None)
    command_queue.put(command)
    robot_controller.join()
    # command_queue.put()