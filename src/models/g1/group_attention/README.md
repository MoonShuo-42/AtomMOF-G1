# G1 组内容与几何偏置组间注意力

本模块对应 H → V(H) → S → S_plus，以及 Z → 37 维几何描述 → 4 头偏置。
所有组内容与几何必须来自同一份 Router 成员权重。
本模块不生成分组；沿用 K_b=min(32,N_b) 和批次级 group_mask。
文中 K 为批次存储宽度 K_batch，实际每个样本的有效组数可以不同。

## 模块化结构

| 文件 / 对象 | 职责 | 可独立修改的部分 |
| --- | --- | --- |
| pooling.py / GroupContentPool | H 投影为 V，再用 Pbar 汇总 S | 内容投影与汇总 |
| geometry.py / GroupGeometryFeatures | 从 Z 提取 37 维标量几何描述 | 几何特征定义 |
| geometry.py / GroupGeometryBias | 37→32→4 的 SiLU MLP | 几何偏置网络 |
| attention.py / GroupSelfAttention | 多头点积加偏置、softmax、读 value 与输出投影 | 组间 attention |
| block.py / GroupAttentionBlock | pre-LN、两次残差与 FFN | 归一化和残差结构 |
| interaction.py / GroupInteraction | 组合上述部分，提供完整与外部 S 两个入口 | 上游连接 |
| utils.py | 输入检查与至少 FP32 的参数运算 | 数值处理 |
| __init__.py | 公共导出 | 对外接口 |

配置：configs/g1/group_attention/default.yaml。
测试：tests/test_group_attention.py。
各子对象目前由构造函数创建，没有任意子模块配置注入机制。

## 1. 内容投影与 S

~~~text
H:     (B,N,256)
V = H W_atom_value: (B,N,128)
Pbar:  (B,K,N)                  # routes.Pbar，组在前
S = Pbar @ V: (B,K,128)
~~~

采用 Linear(256,128,bias=False)，对应研究方案的 H W_v。
该原子内容投影与后续组 attention 的 Q/K/V 投影是独立参数。
Pbar 已包含组内质量归一化与 eps_pool；此处不再按 N、K 或权重总和归一化。
GroupContentPool 返回 V 与 S，便于单独检查投影/池化。

## 2. 从 Z 构造 37 维几何描述

Z 的两条组轴为来源组 r、目标组 k；每个位置最后一维仍为
[r(6), v(3), Q_xx,Q_yy,Q_zz,Q_xy,Q_xz,Q_yz, D] 共 16 通道。
复用已有 src.models.g1.geometry.scalarize_geometry 和 unpack_second_moment。

~~~text
chi(B) = [r(6), D, |v|², tr(Q), tr(Q²), v^T Q v]    # 11 维

v = v(Z_rk), Q = Q(Z_rk)
Qr = Q(Z_rr), Qk = Q(Z_kk)

descriptor_rk = concat(
    chi(Z_rk),                  # 0:11，组间关系
    chi(Z_rr),                  # 11:22，来源组内部
    chi(Z_kk),                  # 22:33，目标组内部
    v^T Qr v,                   # 33，关系方向与来源组
    v^T Qk v,                   # 34，关系方向与目标组
    tr(Q Qr),                   # 35，关系二阶矩与来源组
    tr(Q Qk),                   # 36，关系二阶矩与目标组
)                               # (B,K,K,37)

bias = Linear_32_to_4(SiLU(Linear_37_to_32(descriptor)))
                                # (B,K,K,4)
~~~

37 维特征是固定数学计算，没有可学习参数；后续两层 Linear 带 bias。
Q 先由 6 个分量恢复为对称 3×3 矩阵，再计算迹和收缩；
不能用两个 6 维向量的普通点积替代 tr(Q1 Q2)，否则非对角项会少算一倍。
Z_rr、Z_kk 的对角提取保持来源/目标的顺序，不混淆两个端点。

## 3. 4 头组间注意力与 S_plus

4 头 × 每头 32 维，合计 group_dim=128。
该结构实现研究方案中“拼接投影后加残差/FFN”的 PatternBlock。
图及方案没有细化 LayerNorm 的位置与 FFN 的宽度，因此本次明确采用以下默认值：

~~~text
S_norm = LayerNorm_1(S)
Qh, Kh, Vh = split_heads(QKV_projection(S_norm))

score[b,h,r,k] = dot(Qh[b,h,r], Kh[b,h,k]) / sqrt(32)
                  + bias[b,r,k,h]
A = softmax_over_valid_target_k(score)           # (B,4,K,K)
T_h = A_h @ V_h
T = OutputProjection(concat_heads(T_h))          # (B,K,128)

S1 = S + T
S_plus = S1 + Linear_512_to_128(
    GELU(Linear_128_to_512(LayerNorm_2(S1)))
)
~~~

LayerNorm 含可学习 affine 参数，eps=1e-6；FFN 比例为 4，默认 128→512→128。
QKV 和 attention 输出投影不带 bias；FFN 两层 Linear 带 bias。
首版使用一层 block，没有时间调制、额外 gate、dropout 或最终 LayerNorm。
这些属于本次明确的实现选择；修改位置是 block.py 和配置。

GroupSelfAttention 本身接收已准备的组特征，只实现 attention；
pre-LN 和残差由 GroupAttentionBlock 负责。因此后续可单独更换其中一部分。
所有可学习参数是新初始化的，不自动加载原 AtomMOF small.ckpt 权重。

## 接口与衔接

~~~python
from src.models.g1.aggregation import GeometryAggregator
from src.models.g1.group_attention import GroupInteraction

# 在上层模型 __init__ 中注册一次；不同桥分别持有各自 GroupInteraction
aggregate_geometry = GeometryAggregator()
group_interaction = GroupInteraction()

# 同一份路由同时用于几何与内容
geo = geometry(x_t, l_t, atom_mask)
routes = router(H, c_cond, geo.rho, t, geo.geometry_ok, atom_mask)
relations = aggregate_geometry(
    geo.E, routes.Pbar, atom_mask, routes.group_mask, geo.geometry_ok
)
result = group_interaction(
    H=H,
    Pbar=routes.Pbar,
    Z=relations.Z,
    atom_mask=atom_mask,
    group_mask=routes.group_mask,
    geometry_ok=geo.geometry_ok,
)

S = result.S                           # (B,K,128)
S_plus = result.S_plus                 # (B,K,128)
bias = result.geometry_bias           # (B,K,K,4)
A = result.attention_weights           # (B,4,K,K)，最后一轴为被读取的组
~~~

输入 H 为 (B,N,256)，Pbar 为 (B,K,N)，Z 为 (B,K,K,16)，
atom_mask 为 (B,N)，group_mask 为 (B,K)。geometry_ok 支持 (B,) / (B,1)，
省略时假设所有样本几何可用。mask 支持 bool 或 0/1。
所有有效浮点输入必须有限，且输入、mask 与参数位于同一设备。

可单独调用数学描述、偏置网络、内容池化，也可以接收外部计算的 S：

~~~python
content = group_interaction.content_pool(H, routes.Pbar, atom_mask, routes.group_mask)
descriptor = group_interaction.geometry_bias.features(
    relations.Z, routes.group_mask, geo.geometry_ok
)  # (B,K,K,37)
result = group_interaction.update_groups(
    content.S, relations.Z, routes.group_mask, geo.geometry_ok
)
~~~

单独使用 update_groups 时不会调用内部 content_pool；上层参数注册/训练路径需与此一致。
模块不验证 Z 是否确实由同一份 Pbar 生成，这一对应关系由调用者保证。

## mask、几何不可用与精度

- 原子 padding 在投影前清零；未启用组及 padding 原子的 Pbar 在汇总前清零。
- Z 的无效组对和不可用几何在标量化/高阶乘法之前清零，可忽略被屏蔽的 NaN/Inf。
- geometry_ok=False 时，MLP 输出后再次清零几何偏置，避免 MLP bias 产生虚假信号。
  S 与内容注意力、残差/FFN 继续工作；几何不可用不等于组内容不可用。
- 注意力同时屏蔽查询组和目标组；有效查询只在有效目标组之间归一化。
  全 padding 查询先使用有限 logits，再将输出权重清零，避免全负无穷 softmax。
- 最终无效组的 S/S_plus 为零；bias 与 A 的无效行/列为零。
  全 padding 批次输出及梯度为有限的零。
- K_b=1 时注意力权重为 1，几何偏置无法改变该单目标 softmax；内容投影、残差/FFN 仍可学习。
- 池化、几何收缩、MLP、attention、LayerNorm 和 FFN 至少使用 FP32。
  双精度输入/参数支持 FP64；外层 AMP 或半精度参数不降低这些计算的精度。
  所有类型转换保留梯度。

## 默认配置

~~~yaml
_target_: src.models.g1.group_attention.GroupInteraction
h_dim: 256
group_dim: 128
heads: 4
bias_hidden_dim: 32
ffn_ratio: 4.0
norm_eps: 1.0e-6
~~~

group_dim 必须能被 heads 整除；key/value 的每头宽度为 group_dim/heads。
37 维几何描述的定义固定，bias_hidden_dim 与 heads 可配置。
本模块没有固定 K 的可学习位置嵌入，动态组数由上游路由控制。

## 验证与范围

~~~bash
python -m pytest tests/test_group_attention.py -q
python -m pytest tests -q
~~~

2026-10-15，本地 PyTorch 2.6.0+cpu：新增组交互测试 35 项通过；
全仓库 175 项测试通过（175 passed in 31.42s）。

验证覆盖手算内容池化与注意力、37 通道的独立 NumPy 核对、Q 非对角项计数、
来源/目标顺序、缩放因子、几何偏置实际影响输出、残差/FFN、动态 mask、
几何不可用、全 padding、原子/组置换、半精度与 CPU autocast、有限差分梯度，
以及 ConditionEncoder→PeriodicGeometry→Router→U/Z→S_plus 的真实连接。

几何旋转测试固定 H/Pbar，将 Z 的向量与二阶矩按共同坐标变换。
37 维描述和偏置在此条件下不变；不能据此声称整个 Router 或完整模型严格等变。
这些描述对 v 和 -v 仍给出相同值，只保留部分轴向关系，不完整保留朝向符号或手性。

当前完成 S 与 S_plus；原子读回见 [readback](../readback/README.md)，六模块统一封装见 [G1Module](../README.md)。多桥调度与 Flow 训练/推理接入均已实现，默认关闭。
本地测试不代表 GPU 性能、8 GPU 训练稳定性或 MOF 生成效果。

六模块统一封装与调用方式见 [G1Module](../README.md)。


当前主干接入、开关、checkpoint 兼容性与 263 项全仓库测试记录见 [G1 总文档第 8 节](../README.md#8-atommof-主干接入与开关)。
