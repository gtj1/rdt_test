from core import *

import pygame
from robotcontrol import logger_init

class JoystickFeeder(Process):
    last_command_time: float
    joystick: pygame.joystick.JoystickType
    state_queue: Queue[RobotRecord]
    command_queue: Queue[RobotCommand]
    deadzone: float
    mapping: dict[str, JoystickInputInfo]
    polling_interval: float
    
    def __init__(self, state_queue: Queue[RobotRecord], command_queue: Queue[RobotCommand]):
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
        latest_robot_record = self.state_queue.get()
        robot_state = latest_robot_record['state']
        while not self.state_queue.empty():
            latest_robot_record = self.state_queue.get()
            latest_robot_state = latest_robot_record['state']
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
        gripper_movement = self.apply_deadzone(gripper_open - gripper_close)
        input_state['gripper'] = gripper_movement

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
            gripper_movement *= max_position * 0.05
            new_position = int(np.clip(original_position + gripper_movement, 0, max_position))
            print("new position:", new_position)
            speed = 255
            torque = 0 # int(max_position * abs(gripper_movement))
        else:
            new_position = -1
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
                


if __name__ == '__main__':
    logger_init()
    replay_path = config['path']['replay']
    save_path = config['path']['save']

    command_queue: Queue[RobotCommand] = Queue()
    record_queue: Queue[RobotRecord] = Queue()

    # replay_feeder = ReplayFeeder(command_queue, replay_path)
    # data_collector = DataCollector(record_queue, save_path)

    print('Start collecting data...')

    try:
        robot_controller = RobotController(command_queue, record_queue)
        robot_controller.start()
        time.sleep(1)
    except Exception as e:
        print(f'Failed to initialize arm controller: {e}')
        exit(1)

    joystick_feeder = JoystickFeeder(record_queue, command_queue)
    joystick_feeder.start()
    
    try:
        while True:
            robot_controller.join(1)
            joystick_feeder.join(1)
    except KeyboardInterrupt:
        print("[INFO] Sending shutdown message")
        command_queue.put(RobotCommand(
            command_type='shutdown', arm_state=None, gripper_state=None
        ))
    finally:
        time.sleep(1)
        print("Main process exit")