为了良好的编程环境，请遵循以下规范：
1. 代码风格：遵循 PEP 8 规范，使用 4 个空格进行缩进。
2. 命名规范：变量、函数等命名使用小写字母和下划线分隔，类名使用 PascalCase。命名尽可能简洁明了。
3. 类型注解：函数参数和返回值请添加类型注解，使用 Python 3 的类型提示。
   - 除非函数返回 None，否则应当注明函数的返回类型
   - 除了 self 以外的函数参数都应当注明类型
   - 使用 dict 传参时，建议使用 typing.TypedDict 来定义字典的结构。
   - 对于常用的写起来较长的类型，可以自定义类型别名，例如：
     ```python
     import numpy as np
     from typing import TypeAlias

     Pose: TypeAlias = tuple[np.ndarray, np.ndarray] # (translation/m, rotation/rad)
     ```
   - 当代码高亮没有给出提示时，建议思考前面是否缺少了类型注解。