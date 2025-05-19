import time
import cv2
import json
import numpy as np
import numpy.typing as npt
# from dataclasses import dataclass
from typing import TypedDict, Literal, TypeAlias
from threading import Thread

# import a wrapped `Queue` so that it supports generic types like `Queue[RobotCommand]`
from queue import Queue

import pyrealsense2 as rs
from jodellSdk.jodellSdkDemo import RgClawControl
from robotcontrol import Auboi5Robot, RobotErrorType

with open('src/robot_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

Vector3: TypeAlias = tuple[float, float, float]  # (x, y, z) in meters
EulerAngle: TypeAlias = tuple[float, float, float]  # (roll, pitch, yaw) in radians

PoseEuler: TypeAlias = tuple[Vector3, EulerAngle]

JointPosition: TypeAlias = tuple[float, float, float, float, float, float]

class ArmState(TypedDict):
    arm_name: str
    joint_position: JointPosition
    end_effector_pose: PoseEuler


class GripperState(TypedDict):
    position: int
    speed: int
    torque: int


class RobotCommand(TypedDict):
    command_type: Literal['record', 'move', 'shutdown']
    # The following fields are only used when command_type is not 'shutdown'
    arm_state: ArmState | None
    gripper_state: GripperState | None


class RobotRecord(TypedDict):
    timestamp: float
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


class RobotController(Thread):
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

    undefined_pose: PoseEuler = ((np.nan,) * 3,) * 2
    undefined_joint_position: JointPosition = (np.nan,) * 6

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
    
    def pose_to_arm_state(self, end_effector_pose: PoseEuler) -> ArmState:
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
            # print(f'Moving arm to joint angles {joint_position}')
            # print(f"joint_radian = {joint_position}")
            self.arm.move_joint(joint_position, issync=False)
            # self.arm.add_waypoint(joint_position)
            return True
        except Exception as e:
            print(f'Failed to move arm to joint angles {joint_position}: {e}')
            return False

    def gripper_move_to(self, gripper_state: GripperState) -> bool:
        position = gripper_state['position']
        speed = gripper_state['speed']
        torque = gripper_state['torque']
        if position < 0 or position > self.gripper_max_position:
            # -1 for no move
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
    
    @staticmethod
    def is_valid_pose(pose: PoseEuler) -> bool:
        return not np.isnan(np.array(pose)).any().item()
    
    @staticmethod
    def is_valid_joint_position(joint_position: JointPosition) -> bool:
        return not np.isnan(np.array(joint_position)).any().item()
    
    @staticmethod
    def create_move_command(
        joint_position: JointPosition | None = None,
        end_effector_pose: PoseEuler | None = None,
        gripper_position: int | None = None,
        gripper_speed: int | None = None,
        gripper_force: int | None = None
    ) -> RobotCommand:
        if joint_position is None:
            joint_position = RobotController.undefined_joint_position
        if end_effector_pose is None:
            end_effector_pose = RobotController.undefined_pose
        if gripper_position is None:
            gripper_position = -1
        if gripper_speed is None:
            gripper_speed = 255
        if gripper_force is None:
            gripper_force = 0
        
        arm_state = ArmState(
            arm_name=config['arm']['name'],
            joint_position=joint_position,
            end_effector_pose=end_effector_pose, #
        )
        gripper_state = GripperState(
            position=gripper_position,
            speed=255,
            torque=0
        )
        return RobotCommand(
            command_type='move', arm_state=arm_state, gripper_state=gripper_state
        )

    def execute(self, command: RobotCommand) -> bool:
        command_type = command['command_type']
        if command_type == 'move':
            if command['arm_state'] is None or command['gripper_state'] is None:
                raise ValueError(
                    'Pose and gripper state must be provided for move command')
            joint_position = command['arm_state']['joint_position']
            end_effector_pose = command['arm_state']['end_effector_pose']
            gripper_state = command['gripper_state']

            move_arm = True
            # ensure that both joint_position and end_effector_pose can be used
            if self.is_valid_pose(end_effector_pose):
                # print(f"Using end effector pose {end_effector_pose}")
                command['arm_state'] = self.pose_to_arm_state(end_effector_pose)
            elif self.is_valid_joint_position(joint_position):
                # print("Using joint position")
                arm_state = self.joint_to_arm_state(joint_position)
                command['arm_state'] = arm_state
            else:
                # No move
                move_arm = False
            
            if gripper_state['position'] == -1:
                # indicating that gripper state is not provided
                move_gripper = False
            else:
                move_gripper = True

            self.last_command = command
            result = True
            if move_arm:
                # t1 = time.time()
                result &= self.arm_move_to(command['arm_state']['joint_position'])
                # print(f"Arm move time: {time.time() - t1}")
            if move_gripper:
                # t2 = time.time()
                result &= self.gripper_move_to(gripper_state)
                # print(f"Gripper move time: {time.time() - t2}")
            # print(f"Move arm: {move_arm}, move gripper: {move_gripper}")
            return result
        
        elif command_type == 'shutdown':
            self.running = False
            if self.record_queue is not None:
                self.record_queue.put(RobotRecord(
                    timestamp=time.time(),
                    state=command,
                    action=command
                ))
            return True
        elif command_type == 'record':
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
            command_type='record',
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
            timestamp=time.time(),
            state=robot_record_state,
            action=robot_record_action
        )

        # print(f"putting {robot_record}")
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

