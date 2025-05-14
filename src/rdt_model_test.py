from core import *


# Read the robot state from the record queue and generate a new command, see RobotCommand for details
def model(robot_state: RobotRecord) -> RobotCommand:
    raise NotImplementedError
