from core import *

import os
from typing import Iterator
from robotcontrol import logger_init

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
        


class DataRecorder(Process):
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
        # assert robot_record['state']['command_type'] == 'state'

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
        try:
            while True:

                # Get record from the queue
                robot_record: RobotRecord = self.record_queue.get()
                # assert latest['state']['command_type'] == 'record' and \
                #     latest['action']['command_type'] == 'command'

                print("save")
                camera_frame = self.camera_collector.shot()
                print("shot")
                self.save(robot_record, camera_frame)
        except KeyboardInterrupt:
            pass
        except Exception:
            pass

        finally:
            self.camera_collector.release()



import grasp_demo

from grasp_demo import move

def record():
    grasp_demo.command_queue.put(
        RobotCommand(
            command_type='record',
            arm_state=None,
            gripper_state=None
        )
    )


if __name__ == '__main__':
    logger_init()
    command_queue = Queue[RobotCommand]()
    record_queue = Queue[RobotRecord]()

    grasp_demo.command_queue = command_queue
    grasp_demo.record_queue = record_queue

    try:
        grasp_demo.robot_controller = RobotController(command_queue, record_queue)

        recorder = DataRecorder(record_queue, './out')

        grasp_demo.robot_controller.start()
        recorder.start()
        time.sleep(5) # wait for the arm to initialize
    except Exception as e:
        print(f'Failed to initialize arm controller: {e}')
        exit(1)
    
    move(end_effector_pose=((-0.36, 0.085, 0.30), (3.14, -1.0, -3.14)), gripper_position=0)
    time.sleep(2)
    record()

    move(end_effector_pose=((-0.48, 0.085, 0.18), (3.14, -1.0, -3.14)), gripper_position=80)
    time.sleep(2)
    record()

    move(end_effector_pose=((-0.40, 0, 0.25), (3.14, -1.0, -3.14)))
    time.sleep(1.5)
    record()

    move(end_effector_pose=((-0.32, -0.087, 0.18), (3.14, -1.0, -3.14)), gripper_position=0)
    time.sleep(2)
    record()

    move(end_effector_pose=((-0.32, -0.087, 0.30), (3.14, -1.0, -3.14)), gripper_position=255)
    time.sleep(2)
    record()
    
    command = RobotCommand(
        command_type='shutdown', arm_state=None, gripper_state=None)
    command_queue.put(command)
    grasp_demo.robot_controller.join()