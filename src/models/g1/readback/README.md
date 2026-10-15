# G1 原子读回

本模块对应图中的原子 readback：原子通过内容匹配、几何偏置和 log P 读取 S_plus，
为每个原子—组对生成 value，再按四头权重汇总为 (B,N,128)。
研究方案后续的 128→256 投影与 H 残差更新单独实现，可独立替换。

沿用现有 Router 的 K_b=min(32,N_b)，不创建或截取模板。
以下 K 表示批次存储宽度；每个样本的有效组数由 group_mask 决定。
P、S_plus、U、Z 必须来自同一份分组。本模块不自动验证这一来源关系。

## 模块化结构

| 文件 / 对象 | 职责 | 可独立修改的部分 |
| --- | --- | --- |
| ../geometry/groups.py / extract_group_internal | 提取 Z 的组轴对角线 Z_kk | 组内几何的来源 |
| geometry.py / ReadbackGeometryFeatures | U 与 Z_kk 转成 24 维不变标量 | 描述定义，纯数学计算 |
| geometry.py / ReadbackGeometryBias | 24→32→4 的 SiLU MLP | 几何偏置网络 |
| attention.py / ReadbackAttention | Query/Key、几何偏置、log P、softmax | 读取权重 |
| values.py / RelationConditionedValue | 原子—组关系条件化 value | 内容与几何融合网络 |
| reduction.py / WeightedReadback | 按组加权并拼接各头 | 无参数汇总 |
| projection.py / AtomResidualUpdate | 128→256，再与 H 相加 | 最终输出投影与残差 |
| readback.py / AtomReadback | 组合上述组件 | 上游连接与返回值 |
| utils.py、__init__.py | 输入检查、精度处理、公共导出 | 公共接口 |

配置：configs/g1/readback/default.yaml；测试：tests/test_atom_readback.py。
各组件有独立入口，由 AtomReadback 构造函数注册；没有额外的任意子模块配置注入机制。
几何描述仅计算一次，供 bias 和 value 共用。所有可学习参数新初始化，
与组 attention 的投影独立，不自动加载原 AtomMOF small.ckpt。

## 1. Z_kk 与 24 维描述

U 为 (B,N,K,16)，Z 为 (B,K,K,16)，通道沿用
[r(6),v(3),Q_xx,Q_yy,Q_zz,Q_xy,Q_xz,Q_yz,D]。
只读取 Z 的对角元素，得到 Z_kk (B,K,16)。
Z 的非对角元素不参与本模块计算，但仍由上游组交互模块使用。

~~~text
chi(B) = [r(6), D, |v|², tr(Q), tr(Q²), v^T Q v]  # 11 维

v = v(U_ik), Q = Q(U_ik), Qk = Q(Z_kk)
zeta_ik = concat(
    chi(U_ik),           # 0:11，原子与目标组的关系
    chi(Z_kk),           # 11:22，目标组内部关系
    v^T Qk v,            # 22，关系方向与组内二阶矩
    tr(Q Qk),            # 23，二阶矩之间的关系
)                        # (B,N,K,24)

bias = Linear_32_to_4(SiLU(Linear_24_to_32(zeta)))
                         # (B,N,K,4)
~~~

复用 src.models.g1.geometry 的 scalarize_geometry 和 unpack_second_moment。
Q 恢复为完整对称 3×3 矩阵后再计算，tr(Q Qk) 的非对角项计数两次。
两层偏置 Linear 均带 bias。geometry_ok=False 时，在 MLP 输出后再次清零，
避免可学习偏置在没有几何时产生虚假几何信号。

## 2. Query、Key 与 log P

~~~text
Q = Linear_256_to_128(H)      -> (B,N,4,32)
K = Linear_128_to_128(S_plus) -> (B,K,4,32)

score[b,h,i,k] = dot(Q[b,i,h], K[b,k,h]) / sqrt(32)
                + bias[b,i,k,h]
                + log(P[b,k,i] + eps_pool)
R = softmax_over_valid_group_k(score)  # (B,4,N,K)
~~~

Query/Key 投影不带 bias。这里使用 routes.P，即每个原子分配到各组的权重；
不能换成组内归一化的 routes.Pbar，否则 log 先验的含义会改变。

接口接收组在前的 P (B,K,N)，与 Router 的实际返回格式一致；图中 P_ik 的排列为 (B,N,K)。
若传入 routes.P_nk，需要先转置回 (B,K,N)。当 N=K 时形状本身无法辨别转置错误。
有效位置的 P 必须有限且非负；此处不会重新归一化或检查其和是否等于 1。
该有限性检查会触发设备同步；本模块单独的 GPU 耗时尚无测量记录。

eps_pool 默认 1e-8，应与 Router 一致。
P=0 的有效组仍有 eps 级先验，可通过其他项获得权重；只有 group_mask 执行硬屏蔽。
每个有效原子、每个头在有效组上单独归一化。

## 3. 每个原子—组对自己的 value

~~~text
hidden_ik = W_a H_i + W_s S_plus_k + W_g zeta_ik + b_a
value_ik = W_o SiLU(hidden_ik) + b_o

H_i:256, S_plus_k:128, zeta_ik:24
等价于 concat(H_i,S_plus_k,zeta_ik):408 -> 128 -> 128
value: (B,N,K,128) -> (B,N,K,4,32)
~~~

第一层采用三个独立无偏置 Linear 加一个共同 hidden_bias，
数学上等价于拼接后的单个 Linear，避免展开 408 维原始拼接张量。
第二层 Linear 含 bias。每个组给不同原子的 value 可以不同，
并非仅将 S_plus 投影后对所有原子重复使用。
geometry_ok=False 时只将 zeta 清零，H 和 S_plus 的内容路径继续工作。

当前仍会产生 (B,N,K,128) 级别的中间张量，训练还会保存相应激活；
没有实现分块 value 或融合汇总，实际显存和吞吐需要后续在 GPU 上验证。

## 4. 四头汇总与原子残差

~~~text
M[b,i,h,:] = sum_k R[b,h,i,k] * value[b,i,k,h,:]  # (B,N,4,32)
readback = concat_heads(M)                        # (B,N,128)
delta_H = Linear_128_to_256(readback)              # (B,N,256)
H_updated = H + delta_H                           # (B,N,256)
~~~

最终投影不带 bias。readback 和 delta_H 是不同返回值，避免将 128 维结果直接加到 256 维 H。
本模块按上述公式实现，没有额外 LayerNorm、FFN、时间调制、dropout 或门控。
若以后更改残差更新，修改 projection.py 即可；只需要读回结果可使用 result.readback。

## 接口与上游连接

在上层模型的 __init__ 中创建并注册一次各组件；不要在每次 forward 时重新初始化。

~~~python
from src.models.g1.readback import AtomReadback

atom_readback = AtomReadback(pool_eps=router.pool_eps)

# forward 中，使用同一份路由构造几何、组内容与读回先验
geo = geometry(x_t, l_t, atom_mask)
routes = router(H, c_cond, geo.rho, t, geo.geometry_ok, atom_mask)
relations = aggregate_geometry(
    geo.E, routes.Pbar, atom_mask, routes.group_mask, geo.geometry_ok
)
groups = group_interaction(
    H, routes.Pbar, relations.Z, atom_mask, routes.group_mask, geo.geometry_ok
)
result = atom_readback(
    H=H,
    S_plus=groups.S_plus,
    U=relations.U,
    Z=relations.Z,
    P=routes.P,                 # 此处用 P；上游池化使用 Pbar
    atom_mask=atom_mask,
    group_mask=routes.group_mask,
    geometry_ok=geo.geometry_ok,
)

result.readback          # (B,N,128)
result.delta_H           # (B,N,256)
result.H_updated         # (B,N,256)
result.geometry_bias     # (B,N,K,4)
result.attention_weights # (B,4,N,K)
~~~

atom_mask 为 (B,N)，group_mask 为 (B,K)，支持 bool 或 0/1；
geometry_ok 支持 (B,) 或 (B,1)，省略表示所有样本几何可用。
输入、mask 和参数必须位于同一设备；有效内容及有效几何必须为有限浮点数。

可以分别检查中间表示与 value：

~~~python
zeta = atom_readback.geometry_features(
    relations.U, relations.Z, atom_mask, routes.group_mask, geo.geometry_ok
)  # (B,N,K,24)
values = atom_readback.value(
    H, groups.S_plus, zeta, atom_mask, routes.group_mask, geo.geometry_ok
)  # (B,N,K,4,32)
~~~

主接口不额外返回较大的 values；但训练反向仍需保留相关激活。
独立使用 WeightedReadback 时，调用者必须预先保证权重和值有限且已正确屏蔽。

## mask、精度与边界情况

- 在几何高阶运算和线性投影前清理 padding；几何不可用时清理 U 与 Z_kk。
  被屏蔽位置的 NaN/Inf 可忽略，梯度为有限的零。
- 无效原子—组对的 bias、value、注意力均为零；输出的原子 padding 也为零。
- 全 padding 批次不会对全负无穷 logits 做 softmax，输出和梯度为有限的零。
- 如果有效原子没有可用组，readback 和 delta_H 为零，保留有效 H。
- K_b=1 时，唯一有效组的每头权重为 1；几何 bias 和 log P 无法改变单目标 softmax，
  但几何仍可通过关系条件化 value 影响输出。
- 几何、MLP、注意力、加权汇总和残差至少使用 FP32；支持 FP64。
  外层 AMP 或半精度模型参数不会降低本模块内部的计算精度，类型转换保留梯度。
- 24 维描述在共同旋转下不变。该结论以固定 H/S_plus/P 为前提，
  不证明完整 Router 或模型等变，也不保证这些标量完整表达方向符号或手性。

## 配置与验证

~~~yaml
_target_: src.models.g1.readback.AtomReadback
h_dim: 256
group_dim: 128
heads: 4
head_dim: 32
bias_hidden_dim: 32
value_hidden_dim: 128
pool_eps: 1.0e-8
~~~

读回宽度为 heads*head_dim，不要求它等于 group_dim 或 h_dim；各投影处理维度转换。
几何描述维度固定为 24，动态 K 不对应任何新增位置嵌入或按 K 调整的参数矩阵。

~~~bash
python -m pytest tests/test_atom_readback.py -q
python -m pytest tests -q
~~~

历史测试结果保留在 Git 对应提交中；当前验证状态统一见 [G1 总文档](../README.md#7-验证与实验状态)。
测试覆盖 24 通道 NumPy 独立核对、Z 对角选择、非对角 Q 计数、log P、各头轴与缩放、
408→128→128 value 等价性、手算加权汇总、残差、动态 mask、几何不可用、NaN/Inf padding、
原子/组置换、共同旋转、半精度/CPU autocast、有限差分梯度，
以及 ConditionEncoder→PeriodicGeometry→Router→U/Z→S_plus→readback 的真实连接与反向传播。

原子读回已通过 G1Bridges 接入 Flow 的四个位置；多桥调度和训练/推理调用已实现，默认关闭。
CPU 数值检查不代表 GPU 性能或训练效果。

六模块统一封装与调用方式见 [G1Module](../README.md)。


当前主干接入、训练命令和验证状态见 [G1 总文档](../README.md)。
