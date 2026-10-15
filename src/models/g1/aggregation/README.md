# G1 原子—组与组—组几何汇总

本模块实现图中的两步纯数学计算：U = E Pbar，Z = Pbar^T U。
矩阵表达式是对 16 个几何通道分别计算，不是把四维张量直接相乘。
两步使用同一份 Router 输出 Pbar，不重新计算分组或归一化。

## 计算与布局

~~~text
输入 E:     (B,N,N,16)
输入 Pbar:  (B,K_batch,N)       # 直接使用 routes.Pbar

U[b,i,k,g] = sum_j E[b,i,j,g] * Pbar[b,k,j]
U: (B,N,K_batch,16)

Z[b,r,k,g] = sum_i Pbar[b,r,i] * U[b,i,k,g]
Z: (B,K_batch,K_batch,16)
~~~

U 保留查询原子 i，只汇总目标组 k 的原子 j；Z 再汇总来源组 r 的原子 i。
因此 Z 的两条组轴依次为来源组 r、目标组 k。Z 直接复用 U，不另算一份原子对关系。

沿用上一模块的动态组数 K_b=min(32,N_b)。批次内 K_batch=max(1,max_b K_b)，
使用 routes.group_mask 排除各样本未启用的槽；本模块不固定为 32 组。
输入布局明确采用 (B,K,N)；如只有 Pbar_nk，需显式 transpose(1,2)，不自动猜测轴。

Pbar 已含组质量归一化及 eps_pool，不再次归一化，也不除以 N 或 K。
极小质量组的 Pbar 总和可能小于 1，这一幅度会原样进入 U，并在 Z 的两个端点各作用一次。

## 16 通道含义

| 索引 | 含义 | 通道数 |
| --- | --- | --- |
| 0:6 | 径向响应 r | 6 |
| 6:9 | 有向一阶矩 v_x,v_y,v_z | 3 |
| 9:15 | 对称二阶矩 Q_xx,Q_yy,Q_zz,Q_xy,Q_xz,Q_yz | 6 |
| 15 | 强度 D | 1 |

保持 PeriodicGeometry 的通道顺序与量纲。所有通道逐一线性汇总，
不混合通道、不提前调用 scalarize_geometry、不重建组中心、不做额外 MLP 或投影。
Z 不整体对称化，也不清空对角线：向量通道包含方向，组内关系仍有意义。
对满足周期成对反向关系的 E，标量/Q/D 通道的 Z 应对称，向量通道应反对称；
这是输入几何与共享权重推导出的性质，不通过强制后处理制造。

U/Z 中的 D 是加权关系强度，不能直接当作真实配位数。
Q 是加权二阶矩，不是保证对应某个完整构件朝向的标签。

## 模块化文件

| 文件 / 对象 | 职责 |
| --- | --- |
| contractions.py / AtomToGroupGeometry | 独立实现 E 到 U 的逐通道加权求和 |
| contractions.py / GroupToGroupGeometry | 独立实现 U 到 Z 的逐通道加权求和 |
| geometry.py / GeometryAggregator | 校验接口、清理无效输入、串联两个算子 |
| geometry.py / GeometryAggregationOutput | 返回 U/Z，并注明各轴含义 |
| __init__.py | 公共导出 |

配置为 configs/g1/aggregation/geometry.yaml；测试为 tests/test_geometry_aggregation.py。
两个低层算子只检查形状、类型和设备，要求输入已清理且有限；
通常使用 GeometryAggregator，它负责 atom_mask、group_mask 和 geometry_ok。
各算子均没有可学习参数；梯度可以沿 U/Z 回到 E 和 Pbar，再回到坐标、晶胞及路由参数。

## 接口示例

~~~python
from src.models.g1.aggregation import GeometryAggregator

# 在上层模型中构造一次
geometry_aggregator = GeometryAggregator()

# geometry 和 router 为已有模块；H/c/t 来自相应上游
geo = geometry(x_t, l_t, atom_mask)
routes = router(H, c_cond, geo.rho, t, geo.geometry_ok, atom_mask)
relations = geometry_aggregator(
    E=geo.E,
    Pbar=routes.Pbar,                 # (B,K_batch,N)，不是 P
    atom_mask=atom_mask,
    group_mask=routes.group_mask,
    geometry_ok=geo.geometry_ok,
)
U = relations.U                     # (B,N,K_batch,16)
Z = relations.Z                     # (B,K_batch,K_batch,16)
~~~

geometry_ok 可省略，表示所有样本几何可用；也支持 (B,1) 形式。
geometry_ok=False 只将该样本 U/Z 清零，不修改 Router 的组 mask 或其他内容分支。
返回值保留原子与组索引；下游带 bias 的映射或 attention 仍需使用原有 mask。
同一次结构状态中 E 可跨桥复用，但若 Pbar 改变，该桥的 U/Z 必须重新计算。

## padding、精度与成本

- 在任何乘加之前，按两条原子轴屏蔽 E，按组和原子轴屏蔽 Pbar。
  padding、未启用组或不可用几何中的 NaN/Inf 不会污染结果。
- 无效原子的 U、无效组的 U，以及 Z 对应的行/列为零。
  全 padding 或全不可用几何的输出和梯度为有限的零。
- 有效输入必须是有限浮点数，并与 mask 位于同一设备；没有静默修复有效数据中的异常。
- 两次收缩至少使用 FP32；任一浮点输入为 FP64 时使用 FP64。
  外层 AMP 不会把收缩降为半精度，类型转换保留梯度。
- 使用两次 einsum，避免构造 (B,N,N,K,16) 外积张量。
  计算复杂度约为 O(B*N*N*K*16 + B*N*K*K*16)。
  E 仍是稠密输入，清理 mask 会产生一份同形状副本；尚未实现稀疏或分块收缩。

## 验证与边界

~~~bash
python -m pytest tests/test_geometry_aggregation.py -q
python -m pytest tests -q
~~~

历史测试结果保留在 Git 对应提交中；当前验证状态统一见 [G1 总文档](../README.md#7-验证与实验状态)。

验证覆盖独立双重求和、手算 U 不同但 Z 相同的例子、方向和对角线保留、
Pbar 缩放对 U/Z 的一次/二次作用、动态组 mask、NaN/Inf padding、
原子/组置换、半精度与 autocast、E/Pbar 有限差分梯度、
共同旋转下的向量/二阶矩变换及上游四模块的真实梯度连接。
旋转性质的验证固定同一份 Pbar，将坐标与晶胞共同旋转；
不能据此推断读取任意 H 的 Router 或整个模型严格旋转不变/等变。

本模块只负责 U/Z 汇总。S 内容投影与组间 attention 已由 ../group_attention/ 独立实现；
原子读回见 [readback](../readback/README.md)；现已通过 G1Bridges 接入 Flow 训练与推理，默认关闭。

六模块统一封装与调用方式见 [G1Module](../README.md)。


当前主干接入、训练命令和验证状态见 [G1 总文档](../README.md)。
