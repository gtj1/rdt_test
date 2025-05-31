# 项目说明

本项目基于 **Aubo E5** 机械臂、**Jodell** 夹爪、**Intel RealSense** 相机和 USB 相机，结合 **RDT-1B** 的 VLA 模型，完成物体抓取任务。

预期实现的主要功能包括：
- 数据采集：通过相机获取目标物体的图像数据。
- 模型推理：利用 VLA 模型进行动作决策。
- 机械臂控制：控制 Aubo E5 机械臂完成抓取动作。

本项目旨在实现高效、精准的自动化抓取流程。

# 数据存储说明

请看 `record_data.py` 中的说明：

```python
np.save(
    os.path.join(self.save_path, f'{self.index}.npy'),
    np.array({
        'robot_record': robot_record,
        'camera_frame': camera_frame,
    }),
    allow_pickle=True
)
```

是按照数据一路记录下来的，记录的位置在 `src/robot_config.json` 中可以调。

- `robot_record`：机械臂的记录数据，类型为 RobotRecord，定义见 `core.py`
- `camera_frame`：相机帧数据，类型为 CameraFrame，定义见 `core.py`

# 数据读取与轨迹跟踪

还没做，`ReplayFeeder` 有待完善，大概想法就是从文件读取数据，然后输入队列，但是很麻烦的事情是上一个动作做完之前我设置不了下一个中间点，这个为还在想解决办法。

<!-- 这个机械臂控制，我真的无语了 😅 -->

# 一般调用流程

见 `grasp_demo.py`，可以尝试运行并查看效果。它需要一个队列作为控制指令输入，如果设置了输出队列则可以同时记录输出，见 `record_data.py`。
