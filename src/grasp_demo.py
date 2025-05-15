from core import *

from robotcontrol import logger_init


def create_and_send_command(
    joint_position: JointPosition | None = None,
    end_effector_pose: PoseEuler | None = None,
    gripper_position: int | None = None,
    gripper_speed: int | None = None,
    gripper_force: int | None = None
):
    command = RobotController.create_move_command(
        joint_position=joint_position,
        end_effector_pose=end_effector_pose,
        gripper_position=gripper_position,
        gripper_speed=gripper_speed,
        gripper_force=gripper_force
    )
    command_queue.put(command)

move = create_and_send_command

if __name__ == '__main__':
    logger_init()
    command_queue = Queue[RobotCommand]()
    record_queue: Queue[RobotRecord]
    
    try:
        # not using the record queue for now
        robot_controller = RobotController(command_queue)
        robot_controller.start()
        time.sleep(5) # wait for the arm to initialize
    except Exception as e:
        print(f'Failed to initialize arm controller: {e}')
        exit(1)
    
    move(end_effector_pose=((-0.36, 0.085, 0.30), (3.14, -1.0, -3.14)), gripper_position=0)
    time.sleep(2)

    move(end_effector_pose=((-0.48, 0.085, 0.18), (3.14, -1.0, -3.14)), gripper_position=80)
    time.sleep(2)

    move(end_effector_pose=((-0.40, 0, 0.25), (3.14, -1.0, -3.14)))
    time.sleep(1.5)

    move(end_effector_pose=((-0.32, -0.087, 0.18), (3.14, -1.0, -3.14)), gripper_position=0)
    time.sleep(2)

    move(end_effector_pose=((-0.32, -0.087, 0.30), (3.14, -1.0, -3.14)), gripper_position=255)
    time.sleep(2)
    
    command = RobotCommand(
        command_type='shutdown', arm_state=None, gripper_state=None)
    command_queue.put(command)
    robot_controller.join()