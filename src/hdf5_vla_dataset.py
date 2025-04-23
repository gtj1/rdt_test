# Originally from: https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/main/data/hdf5_vla_dataset.py
# We don't use hdf5 here, but keep its name for compatibility.

import os
import fnmatch
import json
import re
import shutil

import h5py
import yaml
import cv2
import numpy as np

from scipy.spatial.transform import Rotation as R
from configs.state_vec import STATE_VEC_IDX_MAPPING

class HDF5VLADataset:
    """
    This class is used to sample episodes from the embodiment dataset
    stored in HDF5.
    """

    def __init__(self):
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        DATA_DIR = "data/datasets/aubo_e5/rdt_data/"
        self.DATASET_NAME = "aubo"
        self.min_steps = 128

        self.record_paths = []
        self.indices = []
        for foldername in os.listdir(DATA_DIR):
            folder_path = os.path.join(DATA_DIR, foldername)
            ret, indices = self.read_npy_indices(folder_path)
            if ret:
                self.record_paths.append(folder_path)
                self.indices.append(indices)
                
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
        
        episode_lens = []
        for file_path, indices in zip(self.record_paths, self.indices):
            episode_lens.append(len(indices))
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    @staticmethod
    def init_states_and_actions(path: str, indices: range):
        states = []
        actions = []
        for index in indices:
            result = np.load(os.path.join(path, f"{index}.npy")).item()['robot_record']
            state = result['state']
            joint = state['arm_state']['joint_position']
            xyz, rpy = state['arm_state']['end_effector_pose']
            rotation_matrix = R.from_euler('xyz', rpy, degrees=True).as_matrix()
            ortho6d = rotation_matrix[:, :2].transpose().flatten()
            gripper_position = [state['gripper_state']['position'] / 255.0]
            state_vec = np.concatenate([joint, xyz, ortho6d, gripper_position])
            states.append(state_vec)

            action = result['action']
            joint = action['arm_action']['joint_position']
            xyz, rpy = action['arm_action']['end_effector_pose']
            rotation_matrix = R.from_euler('xyz', rpy, degrees=True).as_matrix()
            ortho6d = rotation_matrix[:, :2].transpose().flatten()
            gripper_position = [action['gripper_action']['position'] / 255.0]
            action_vec = np.concatenate([joint, xyz, ortho6d, gripper_position])
            actions.append(action_vec)
        
        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        np.save(os.path.join(path, "states.npy"), states)
        np.save(os.path.join(path, "actions.npy"), actions)

    @classmethod
    def read_npy_indices(cls, path: str, pattern: str | re.Pattern = r'^(\d+)\.npy$') -> tuple[bool, range | str]:
        """
        Check if the folder contains valid data. If so, change to standard format,
        extract pose information, and return the indices of the episodes.
        """
        if not os.path.exists(path):
            return (False, "Non exist")
        if not os.path.isdir(path):
            return (False, "Isn't a directory")
        if not os.path.exists(os.path.join(path, 'instruction.json')):
            return (False, "Instruction not specified")
        npy_files: list[str] = []
        for filename in os.listdir(path):
            if filename.endswith('.npy'):
                npy_files.append(filename)
        if len(npy_files) == 0:
            return (False, "No npy file found")
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        matches = [pattern.match(filename) for filename in npy_files]
        indices = [int(filename_match.group(1)) for filename_match in matches if filename_match]
        if not indices:
            return (False, "Files found but not in expected format")
        # check if the indices are continuous
        expected_range = range(min(indices), max(indices) + 1)
        if len(set(indices)) != len(expected_range):
            return (False, "Discontinuous Indices")
        for filename_match in matches:
            index_str = filename_match.group(1)
            new_index_str = f"{int(index_str)}.npy"
            if index_str != new_index_str:
                original_name = os.path.join(path, filename_match.group(0))
                new_name = os.path.join(path, new_index_str)
                os.rename(original_name, new_name)
        if os.path.exists(os.path.join(path, "states.npy")):
            cls.init_states_and_actions(path, expected_range)
        return (True, expected_range)
    
    def __len__(self):
        return len(self.record_paths)

    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionar
        """
        while True:
            if index is None:
                record_path = np.random.choice(self.record_paths, p=self.episode_sample_weights)
            else:
                record_path = self.record_paths[index]
            valid, sample = self.parse_npy_file(record_path) \
                if not state_only else self.parse_npy_file_state_only(record_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.record_paths))
    
    def parse_npy_file(self, record_path: str):
        """[Modify] Parse a npy file to generate a training sample at
            a random timestep.

        Args:
            record_path (str): the path to the npy file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        ret, record_indices = self.read_npy_indices(record_path)
        if not ret:
            return False, None
        states: np.ndarray = np.load(os.path.join(record_path, "states.npy"))
        actions: np.ndarray = np.load(os.path.join(record_path, "actions.npy"))
        num_steps = states.shape[0]
        # [Optional] We drop too-short episode
        if num_steps < self.min_steps:
            return False, None
        
        # [Optional] We skip the first few still steps
        EPS = 1e-2
        # Get the idx of the first qpos whose delta exceeds the threshold
        state_delta = np.abs(states - states[0])
        indices = np.where(np.any(state_delta > EPS, axis=1))[0]
        if len(indices) == 0:
                raise ValueError("Found no state vector that exceeds the threshold.")
        first_idx = indices[0]

        # We randomly sample a timestep
        step_id = np.random.randint(first_idx - 1, num_steps) + record_indices.start

        # Load the instruction
        with open(os.path.join(record_path, 'instruction.json'), 'r') as f_instr:
            instruction_dict = json.load(f_instr)
        # We have 1/3 prob to use original instruction,
        # 1/3 to use simplified instruction,
        # and 1/3 to use expanded instruction.
        instruction_type = np.random.choice([
            'instruction', 'simplified_instruction', 'expanded_instruction'])
        instruction = instruction_dict[instruction_type]
        if isinstance(instruction, list):
            instruction = np.random.choice(instruction)
        # You can also use precomputed language embeddings (recommended)
        # instruction = "path/to/lang_embed.pt"
        
        # Assemble the meta
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": instruction
        }
        start_index = step_id - record_indices.start
        action_end_index = start_index + self.CHUNK_SIZE
        current_state = states[start_index: start_index + 1]
        state_std = np.std(states, axis=0)
        state_mean = np.mean(states, axis=0)
        state_norm = np.linalg.norm(states, axis=0)
        target_actions = actions[start_index:action_end_index]
        if actions.shape[0] < self.CHUNK_SIZE:
            target_actions = np.concatenate([
                actions,
                np.tile(actions[-1], (self.CHUNK_SIZE - actions.shape[0], 1))
            ], axis=0)
        
        # Fill the state/action into the unified vector
        def fill_in_state(values):
            # Target indices corresponding to your state space
            # [joint, xyz, ortho6d, gripper_position]
            UNI_STATE_INDICES = [
                STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(6)
            ] + [
                STATE_VEC_IDX_MAPPING[f"eff_pose_{ax}"] for ax in "xyz"
            ] + [
                STATE_VEC_IDX_MAPPING[f"eff_angle_{i}"] for i in range(6)
            ] + [
                STATE_VEC_IDX_MAPPING[f"gripper_joint_0_pos"]
            ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec
        current_state = fill_in_state(current_state)
        state_indicator = fill_in_state(np.ones_like(current_state))
        state_std = fill_in_state(state_std)
        state_mean = fill_in_state(state_mean)
        state_norm = fill_in_state(state_norm)
        # If action's format is different from state's,
        # you may implement fill_in_action()

        target_actions = fill_in_state(target_actions)

        # Parse the images
        def parse_image(keys: list[str]) -> dict[str, np.ndarray]:
            imgs = {key: [] for key in keys}
            for i in range(max(step_id - self.IMG_HISORY_SIZE + 1, 0), step_id + 1):
                npy_filename = os.path.join(record_path, f"{i}.npy")
                camera_frame = np.load(npy_filename)['camera_frame']
                for key in keys:
                    img = camera_frame[key]
                    imgs[key].append(img)
            for key in keys:
                array = np.stack(imgs[key])
                if array.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    array = np.concatenate([
                        np.tile(array[0], (self.IMG_HISORY_SIZE - array.shape[0], 1, 1, 1)),
                        array
                    ], axis=0)
                imgs[key] = array
            return imgs

        img_keys = ['rgbd_image', 'usb_image']
        imgs = parse_image(self.img_keys)
        valid_len = min(step_id - start_index - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
        valid_mask = np.array(
            [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
        )
        false_mask = np.array([False] * self.IMG_HISORY_SIZE)
        cam_right_wrist = imgs['rgbd_image']
        cam_high = imgs['usb_image']
        # No left wrist here
        cam_left_wrist = np.zeros_like(cam_right_wrist)

        cam_left_wrist_mask = false_mask
        cam_right_wrist_mask = valid_mask
        cam_high_mask = valid_mask.copy()

        # Return the resulting sample
        # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
        # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
        # if the left-wrist camera is unavailable on your robot
        return True, {
            "meta": meta,
            "state": current_state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "actions": actions,
            "state_indicator": state_indicator,
            "cam_high": cam_high,
            "cam_high_mask": cam_high_mask,
            "cam_left_wrist": cam_left_wrist,
            "cam_left_wrist_mask": false_mask,
            "cam_right_wrist": cam_right_wrist,
            "cam_right_wrist_mask": cam_right_wrist_mask
        }
        
    
    def parse_npy_file_state_only(self, record_path):
        """[Modify] Parse a npy file to generate a training sample at
            a random timestep.

        Args:
            record_path (str): the path to the npy file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        ret, record_indices = self.read_npy_indices(record_path)
        if not ret:
            return False, None
        states: np.ndarray = np.load(os.path.join(record_path, "states.npy"))
        actions: np.ndarray = np.load(os.path.join(record_path, "actions.npy"))
        num_steps = states.shape[0]
        # [Optional] We drop too-short episode
        if num_steps < self.min_steps:
            return False, None
        
        # [Optional] We skip the first few still steps
        EPS = 1e-2
        # Get the idx of the first qpos whose delta exceeds the threshold
        state_delta = np.abs(states - states[0])
        indices = np.where(np.any(state_delta > EPS, axis=1))[0]
        if len(indices) == 0:
                raise ValueError("Found no state vector that exceeds the threshold.")
        first_idx = indices[0]

        # We randomly sample a timestep
        step_id = np.random.randint(first_idx - 1, num_steps) + record_indices.start

        # Load the instruction
        with open(os.path.join(record_path, 'instruction.json'), 'r') as f_instr:
            instruction_dict = json.load(f_instr)
        # We have 1/3 prob to use original instruction,
        # 1/3 to use simplified instruction,
        # and 1/3 to use expanded instruction.
        instruction_type = np.random.choice([
            'instruction', 'simplified_instruction', 'expanded_instruction'])
        instruction = instruction_dict[instruction_type]
        if isinstance(instruction, list):
            instruction = np.random.choice(instruction)
        # You can also use precomputed language embeddings (recommended)
        # instruction = "path/to/lang_embed.pt"
        
        # Assemble the meta
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": instruction
        }
        start_index = step_id - record_indices.start
        action_end_index = start_index + self.CHUNK_SIZE
        current_state = states[start_index: start_index + 1]
        state_std = np.std(states, axis=0)
        state_mean = np.mean(states, axis=0)
        state_norm = np.linalg.norm(states, axis=0)
        target_actions = actions[start_index:action_end_index]
        if actions.shape[0] < self.CHUNK_SIZE:
            target_actions = np.concatenate([
                actions,
                np.tile(actions[-1], (self.CHUNK_SIZE - actions.shape[0], 1))
            ], axis=0)
        
        # Fill the state/action into the unified vector
        def fill_in_state(values):
            # Target indices corresponding to your state space
            # [joint, xyz, ortho6d, gripper_position]
            UNI_STATE_INDICES = [
                STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(6)
            ] + [
                STATE_VEC_IDX_MAPPING[f"eff_pose_{ax}"] for ax in "xyz"
            ] + [
                STATE_VEC_IDX_MAPPING[f"eff_angle_{i}"] for i in range(6)
            ] + [
                STATE_VEC_IDX_MAPPING[f"gripper_joint_0_pos"]
            ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec
        current_state = fill_in_state(current_state)
        state_indicator = fill_in_state(np.ones_like(current_state))
        state_std = fill_in_state(state_std)
        state_mean = fill_in_state(state_mean)
        state_norm = fill_in_state(state_norm)
        # If action's format is different from state's,
        # you may implement fill_in_action()

        target_actions = fill_in_state(target_actions)

        return True, {
            "state": current_state,
            "action": target_actions,
        }
        
        
# class HDF5VLADataset:
#     """
#     This class is used to sample episodes from the embododiment dataset
#     stored in HDF5.
#     """
#     def __init__(self) -> None:
#         # [Modify] The path to the HDF5 dataset directory
#         # Each HDF5 file contains one episode
#         HDF5_DIR = "data/datasets/agilex/rdt_data/"
#         self.DATASET_NAME = "agilex"
        
#         self.file_paths = []
#         for root, _, files in os.walk(HDF5_DIR):
#             for filename in fnmatch.filter(files, '*.hdf5'):
#                 file_path = os.path.join(root, filename)
#                 self.file_paths.append(file_path)
                
#         # Load the config
#         with open('configs/base.yaml', 'r') as file:
#             config = yaml.safe_load(file)
#         self.CHUNK_SIZE = config['common']['action_chunk_size']
#         self.IMG_HISORY_SIZE = config['common']['img_history_size']
#         self.STATE_DIM = config['common']['state_dim']
    
#         # Get each episode's len
#         episode_lens = []
#         for file_path in self.file_paths:
#             valid, res = self.parse_hdf5_file_state_only(file_path)
#             _len = res['state'].shape[0] if valid else 0
#             episode_lens.append(_len)
#         self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
#     def __len__(self):
#         return len(self.file_paths)
    
#     def get_dataset_name(self):
#         return self.DATASET_NAME
    
#     def get_item(self, index: int=None, state_only=False):
#         """Get a training sample at a random timestep.

#         Args:
#             index (int, optional): the index of the episode.
#                 If not provided, a random episode will be selected.
#             state_only (bool, optional): Whether to return only the state.
#                 In this way, the sample will contain a complete trajectory rather
#                 than a single timestep. Defaults to False.

#         Returns:
#            sample (dict): a dictionary containing the training sample.
#         """
#         while True:
#             if index is None:
#                 file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
#             else:
#                 file_path = self.file_paths[index]
#             valid, sample = self.parse_hdf5_file(file_path) \
#                 if not state_only else self.parse_hdf5_file_state_only(file_path)
#             if valid:
#                 return sample
#             else:
#                 index = np.random.randint(0, len(self.file_paths))
    
#     def parse_hdf5_file(self, file_path):
#         """[Modify] Parse a hdf5 file to generate a training sample at
#             a random timestep.

#         Args:
#             file_path (str): the path to the hdf5 file
        
#         Returns:
#             valid (bool): whether the episode is valid, which is useful for filtering.
#                 If False, this episode will be dropped.
#             dict: a dictionary containing the training sample,
#                 {
#                     "meta": {
#                         "dataset_name": str,    # the name of your dataset.
#                         "#steps": int,          # the number of steps in the episode,
#                                                 # also the total timesteps.
#                         "instruction": str      # the language instruction for this episode.
#                     },                           
#                     "step_id": int,             # the index of the sampled step,
#                                                 # also the timestep t.
#                     "state": ndarray,           # state[t], (1, STATE_DIM).
#                     "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
#                     "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
#                     "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
#                     "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
#                     "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
#                     "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
#                                                 # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
#                     "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
#                                                 # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
#                     "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
#                                                 # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
#                     "cam_left_wrist_mask": ndarray,
#                     "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
#                                                 # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
#                                                 # If only one wrist, make it right wrist, plz.
#                     "cam_right_wrist_mask": ndarray
#                 } or None if the episode is invalid.
#         """
#         with h5py.File(file_path, 'r') as f:
#             qpos = f['observations']['qpos'][:]
#             num_steps = qpos.shape[0]
#             # [Optional] We drop too-short episode
#             if num_steps < 128:
#                 return False, None
            
#             # [Optional] We skip the first few still steps
#             EPS = 1e-2
#             # Get the idx of the first qpos whose delta exceeds the threshold
#             qpos_delta = np.abs(qpos - qpos[0:1])
#             indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
#             if len(indices) > 0:
#                 first_idx = indices[0]
#             else:
#                 raise ValueError("Found no qpos that exceeds the threshold.")
            
#             # We randomly sample a timestep
#             step_id = np.random.randint(first_idx-1, num_steps)
            
#             # Load the instruction
#             dir_path = os.path.dirname(file_path)
#             with open(os.path.join(dir_path, 'expanded_instruction_gpt-4-turbo.json'), 'r') as f_instr:
#                 instruction_dict = json.load(f_instr)
#             # We have 1/3 prob to use original instruction,
#             # 1/3 to use simplified instruction,
#             # and 1/3 to use expanded instruction.
#             instruction_type = np.random.choice([
#                 'instruction', 'simplified_instruction', 'expanded_instruction'])
#             instruction = instruction_dict[instruction_type]
#             if isinstance(instruction, list):
#                 instruction = np.random.choice(instruction)
#             # You can also use precomputed language embeddings (recommended)
#             # instruction = "path/to/lang_embed.pt"
            
#             # Assemble the meta
#             meta = {
#                 "dataset_name": self.DATASET_NAME,
#                 "#steps": num_steps,
#                 "step_id": step_id,
#                 "instruction": instruction
#             }
            
#             # Rescale gripper to [0, 1]
#             qpos = qpos / np.array(
#                [[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]] 
#             )
#             target_qpos = f['action'][step_id:step_id+self.CHUNK_SIZE] / np.array(
#                [[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]] 
#             )
            
#             # Parse the state and action
#             state = qpos[step_id:step_id+1]
#             state_std = np.std(qpos, axis=0)
#             state_mean = np.mean(qpos, axis=0)
#             state_norm = np.sqrt(np.mean(qpos**2, axis=0))
#             actions = target_qpos
#             if actions.shape[0] < self.CHUNK_SIZE:
#                 # Pad the actions using the last action
#                 actions = np.concatenate([
#                     actions,
#                     np.tile(actions[-1:], (self.CHUNK_SIZE-actions.shape[0], 1))
#                 ], axis=0)
            
#             # Fill the state/action into the unified vector
#             def fill_in_state(values):
#                 # Target indices corresponding to your state space
#                 # In this example: 6 joints + 1 gripper for each arm
#                 UNI_STATE_INDICES = [
#                     STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
#                 ] + [
#                     STATE_VEC_IDX_MAPPING["left_gripper_open"]
#                 ] + [
#                     STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
#                 ] + [
#                     STATE_VEC_IDX_MAPPING["right_gripper_open"]
#                 ]
#                 uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
#                 uni_vec[..., UNI_STATE_INDICES] = values
#                 return uni_vec
#             state = fill_in_state(state)
#             state_indicator = fill_in_state(np.ones_like(state_std))
#             state_std = fill_in_state(state_std)
#             state_mean = fill_in_state(state_mean)
#             state_norm = fill_in_state(state_norm)
#             # If action's format is different from state's,
#             # you may implement fill_in_action()
#             actions = fill_in_state(actions)
            
#             # Parse the images
#             def parse_img(key):
#                 imgs = []
#                 for i in range(max(step_id-self.IMG_HISORY_SIZE+1, 0), step_id+1):
#                     img = f['observations']['images'][key][i]
#                     imgs.append(cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR))
#                 imgs = np.stack(imgs)
#                 if imgs.shape[0] < self.IMG_HISORY_SIZE:
#                     # Pad the images using the first image
#                     imgs = np.concatenate([
#                         np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
#                         imgs
#                     ], axis=0)
#                 return imgs
#             # `cam_high` is the external camera image
#             cam_high = parse_img('cam_high')
#             # For step_id = first_idx - 1, the valid_len should be one
#             valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
#             cam_high_mask = np.array(
#                 [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
#             )
#             cam_left_wrist = parse_img('cam_left_wrist')
#             cam_left_wrist_mask = cam_high_mask.copy()
#             cam_right_wrist = parse_img('cam_right_wrist')
#             cam_right_wrist_mask = cam_high_mask.copy()
            
#             # Return the resulting sample
#             # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
#             # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
#             # if the left-wrist camera is unavailable on your robot
#             return True, {
#                 "meta": meta,
#                 "state": state,
#                 "state_std": state_std,
#                 "state_mean": state_mean,
#                 "state_norm": state_norm,
#                 "actions": actions,
#                 "state_indicator": state_indicator,
#                 "cam_high": cam_high,
#                 "cam_high_mask": cam_high_mask,
#                 "cam_left_wrist": cam_left_wrist,
#                 "cam_left_wrist_mask": cam_left_wrist_mask,
#                 "cam_right_wrist": cam_right_wrist,
#                 "cam_right_wrist_mask": cam_right_wrist_mask
#             }

#     def parse_hdf5_file_state_only(self, file_path):
#         """[Modify] Parse a hdf5 file to generate a state trajectory.

#         Args:
#             file_path (str): the path to the hdf5 file
        
#         Returns:
#             valid (bool): whether the episode is valid, which is useful for filtering.
#                 If False, this episode will be dropped.
#             dict: a dictionary containing the training sample,
#                 {
#                     "state": ndarray,           # state[:], (T, STATE_DIM).
#                     "action": ndarray,          # action[:], (T, STATE_DIM).
#                 } or None if the episode is invalid.
#         """
#         with h5py.File(file_path, 'r') as f:
#             qpos = f['observations']['qpos'][:]
#             num_steps = qpos.shape[0]
#             # [Optional] We drop too-short episode
#             if num_steps < 128:
#                 return False, None
            
#             # [Optional] We skip the first few still steps
#             EPS = 1e-2
#             # Get the idx of the first qpos whose delta exceeds the threshold
#             qpos_delta = np.abs(qpos - qpos[0:1])
#             indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
#             if len(indices) > 0:
#                 first_idx = indices[0]
#             else:
#                 raise ValueError("Found no qpos that exceeds the threshold.")
            
#             # Rescale gripper to [0, 1]
#             qpos = qpos / np.array(
#                [[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]] 
#             )
#             target_qpos = f['action'][:] / np.array(
#                [[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]] 
#             )
            
#             # Parse the state and action
#             state = qpos[first_idx-1:]
#             action = target_qpos[first_idx-1:]
            
#             # Fill the state/action into the unified vector
#             def fill_in_state(values):
#                 # Target indices corresponding to your state space
#                 # In this example: 6 joints + 1 gripper for each arm
#                 UNI_STATE_INDICES = [
#                     STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
#                 ] + [
#                     STATE_VEC_IDX_MAPPING["left_gripper_open"]
#                 ] + [
#                     STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
#                 ] + [
#                     STATE_VEC_IDX_MAPPING["right_gripper_open"]
#                 ]
#                 uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
#                 uni_vec[..., UNI_STATE_INDICES] = values
#                 return uni_vec
#             state = fill_in_state(state)
#             action = fill_in_state(action)
            
#             # Return the resulting sample
#             return True, {
#                 "state": state,
#                 "action": action
#             }

if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)