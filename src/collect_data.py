from core import *

import os
from typing import Iterator

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

