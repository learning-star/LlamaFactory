# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
EcoPhase - 梯度监控和 ECS 通信模块（gRPC 版本）

通信版本：
- EcoMonitor：梯度监控和 ECS 通信模块（简化配置版本）
- 所有核心决策（LR调整、早停）都由 ECS 服务端完成
- 使用 gRPC 进行高效通信，支持 TLS 加密
"""

import sys
import types

# ============================================================
# Nuitka 推荐的运行时模块保护
# 限制外部只能访问 __all__ 中定义的接口
# ============================================================

__all__ = ("EcoMonitor",)


def __dir__():
    """限制 dir() 只返回公开接口"""
    return list(__all__)


class _ProtectedModule(types.ModuleType):
    """
    保护模块：只允许访问 __all__ 中定义的属性
    
    这样即使代码被反编译，攻击者也无法访问内部实现细节
    """
    
    def __init__(self, name):
        types.ModuleType.__init__(self, name)
        self.__dict__.update(globals())
    
    def __getattr__(self, name):
        """拦截属性访问，只允许访问公开接口"""
        if name in __all__:
            # 允许访问公开接口
            return self.__dict__.get(name)
        
        if name in ("__path__", "__file__", "__doc__", "__name__", "__package__", "__version__"):
            # 允许访问元信息
            return self.__dict__.get(name)
        
        # 拒绝访问内部属性
        raise AttributeError(
            f"module '{self.__name__}' has no attribute '{name}'. "
            f"Available interfaces: {list(__all__)}"
        )
    
    def __dir__(self):
        """限制 dir() 输出"""
        return list(__all__)


# ============================================================
# 导入公开接口（在替换模块对象之前）
# ============================================================
from .EcoMonitor import EcoMonitor  # noqa: E402


# ============================================================
# 替换当前模块为保护版本
# 这必须在所有导入之后进行
# ============================================================
sys.modules[__name__] = _ProtectedModule(__name__)

