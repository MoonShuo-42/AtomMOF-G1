# G1 随原子数变化的模板与软组路由

从当前原子隐藏特征 H、条件特征 c、几何摘要 ρ、时间 t 和几何可用标志，
计算 P、Pbar 及每组软质量 mass。每个样本的有效模板数和软组数统一为：

~~~text
N_b = sum_i atom_mask[b,i]
K_b = min(max_groups, N_b)    # 默认 max_groups=32
~~~

例如有效原子数 1、10、20、32、40 分别启用 1、10、20、32、32 个模板和软组。
N_b 使用 mask 统计，不使用 padding 后的张量长度；空样本 K_b=0。
这次修改替换此前“每个样本全部启用 32 个模板和软组”的实现。

参数库仍保留最多 32 行，每个样本启用前 K_b 行。模板在构造时初始化一次，
不在 forward 中增删或重建 Parameter，因此优化器状态可以持续保留。
变化的是参与该样本路由的模板数及软组数；单个模板和 MLP 输出的特征维度仍为 32。

此前讨论的前缀选择问题仍然存在：靠后的模板只从足够大的样本获得任务梯度，
不同模板的训练机会和数据分布可能不同。本次按原子数修改并未解决该训练问题，
也没有额外引入模板生成器、随机选择或均衡损失。

## 模块化结构

| 文件 / 对象 | 职责 | 后续修改位置 |
| --- | --- | --- |
| time.py / FourierTimeEmbedding | 固定 32 维正弦余弦时间特征 | 改变时间频率或编码定义 |
| features.py / RouterQueryEncoder | 拼接、padding/几何可用性处理、MLP 和 LayerNorm | 增减条件、修改融合网络 |
| groups.py / AtomCountGroups | 根据有效原子数计算 K 和前缀 group_mask | 修改组数规则或选择策略 |
| templates.py / TemplateBank | 持有最大模板参数库，计算批次所需前缀的缩放点积 | 改变模板初始化或打分方式 |
| normalization.py / normalize_assignments | 排除无效组，沿组 softmax、组质量和 Pbar 归一化 | 修改归一化策略 |
| normalization.py / RoutingOutput | 明确输出布局与转置视图 | 后续 pooling / 几何汇总接口 |
| router.py / SoftRouter | 组合上述部分，提供时间标量/预编码两个入口 | 上层接入或组合 |
| __init__.py | 公共导出 | 对外接口 |

配置：configs/g1/router/adaptive.yaml。
测试：tests/test_soft_router.py。

沿用职责拆分和固定接口；子对象目前由 SoftRouter 构造函数创建，
没有任意子模块配置注入机制。各部分可以独立修改和验证。
本模块负责路由，已通过 G1Bridges 接入 Flow 主干。U/Z 几何汇总由 ../aggregation/ 独立实现；
S 内容汇总与组间更新见 ../group_attention/；原子读回见 ../readback/。

## 接口和布局

~~~python
from src.models.g1.routing import SoftRouter

router = SoftRouter()  # 在上层模型中构造一次并注册为子模块
routes = router(
    h=H,
    c=c_cond,
    rho=geo.rho,
    t=t,
    geometry_ok=geo.geometry_ok,
    atom_mask=atom_mask,
)

P = routes.P                    # (B, K_batch, N)
Pbar = routes.Pbar              # (B, K_batch, N)
mass = routes.mass              # (B, K_batch)
group_mask = routes.group_mask  # (B, K_batch)，有效模板/软组
K = routes.group_counts         # (B,)，每个样本的 K_b

# 如后续算子使用原方案的 (B,N,K) 写法：
P_nk = routes.P_nk        # (B, N, K_batch)，转置视图，不复制
Pbar_nk = routes.Pbar_nk  # (B, N, K_batch)，转置视图，不复制
~~~

对外默认使用图中偏好的 (B,K,N) 布局。批次内使用
K_batch=max(1,max_b K_b) 对齐存储，每个样本不足的组通过 group_mask 排除。
全 padding 批次保留一列全零存储槽，但实际组数仍为 0。
读取批次最大 K 会在 CUDA 上产生一次主机同步；这是当前紧凑输出布局的代价。
内部打分先生成 (B,N,K_batch)，未启用的组在 softmax 前置为负无穷，
最后沿有效模板做 softmax 并转置输出。因此：

- routes.P.sum(dim=1)：沿组求和，对每个有效原子为 1。
- routes.mass = routes.P.sum(dim=-1)：沿原子求和，得到每组软质量。
- 无效组的 P、Pbar、mass 严格为零；group_counts 不从 mass 大小推断。
- routes.Pbar = routes.P / (routes.mass[..., None] + eps)。
- routes.Pbar.sum(dim=-1) = mass / (mass + eps)，只在正常非空组时近似 1。
- 输出及视图不保证 contiguous；只有具体后续算子需要时才调用 contiguous()。

例如后续有原子 value V，形状 (B,N,D)，可用 routes.Pbar @ V 得到 (B,K_batch,D)。
后续组 attention、带 bias 的投影和损失计算也需继续使用 group_mask，避免 padding 组参与更新。
这里只说明布局用途，本模块未定义 value 投影；U/Z 汇总接口见 ../aggregation/README.md。

| 输入 | 默认形状 | 含义 |
| --- | --- | --- |
| H / h | (B,N,256) | 当前原子隐藏特征 |
| c | (B,N,128) | ConditionEncoder 输出 |
| rho | (B,N,11) | PeriodicGeometry 输出的标量摘要 |
| t | (B,) 或 (B,1) | 当前生成时间，正常输入为 0 到 1，不在模块内裁剪 |
| geometry_ok | (B,) 或 (B,1) | 该结构几何是否可用；bool 或 0/1 |
| atom_mask | (B,N) | 有效原子标志；bool 或 0/1 |

输入和模型参数须在同一设备。有效 H/c/时间、以及可用几何的 rho 应为有限数值；
padding 及明确不可用的 rho 在进入 MLP 前清零，可安全忽略其中的 NaN/Inf。
输入不涉及构件编号，不施加 ConditionEncoder 的同构件 attention 范围限制。

## 计算定义

### 固定时间编码

默认用 16 对频率，总计 32 维，按每对 sin/cos 交错排列：

~~~text
omega_p = 2*pi*2^(p/4), p=0,...,15

gamma(t) = [
    sin(omega_0*t), cos(omega_0*t),
    ...
    sin(omega_15*t), cos(omega_15*t)
]
~~~

此处复现研究方案的固定编码，不调用原 DiT 的可学习 TimestepEmbedder。
频率没有可训练参数；时间本身可正常获得梯度。

如多座桥共享同一时间编码，可使用缓存入口：

~~~python
t_emb = router.time_embedding(t)  # (B,32)
routes = router.route_with_time_embedding(
    H, c_cond, geo.rho, t_emb, geo.geometry_ok, atom_mask
)
~~~

这个入口接受符合上述频率和排列约定的 (B,32) 张量，避免重复计算时间特征。
条件/几何/时间相同也不意味着 P 可跨桥复用：H 和各桥参数可能不同。

### 特征融合和模板点积

~~~text
z_i = concat(h_i, c_i, rho_i, gamma(t), g_ok)
    : 256 + 128 + 11 + 32 + 1 = 428

a_i = LayerNorm(Linear_128_to_32(SiLU(Linear_428_to_128(z_i))))

template_bank: (max_groups,32)
K_b = min(max_groups, N_b)
s_bik = a_bi^T templates_k / sqrt(32), k < K_b
P_bik = softmax_{k<K_b}(s_bik)
P_bik = 0, k >= K_b 或原子无效
~~~

MLP 的两个 Linear 带 bias，LayerNorm 带可学习 affine 参数。
模板初始化为标准正态分布，之后作为 nn.Parameter 参与优化。
打分不额外做余弦归一化或按分数 top-k 选择，使用随有效原子数变化的前缀 mask。

max_groups 是最大容量，默认 32；每个样本的 K 自动由 atom_mask 计算。
配置已从 fixed32.yaml 更名为 adaptive.yaml，SoftRouter 构造参数
num_templates 已改名为 max_groups，以区分容量和实际参与路由的数量。
一个 SoftRouter 实例持有一套模板；需要每桥独立参数时，为每座桥分别实例化。

### Pbar 的归一化

以下写法用原子在前的 P_nk 表示：

~~~text
m_k = sum_{i valid} P_ik
Pbar_ik = P_ik / (m_k + eps_pool)
eps_pool = 1e-8
~~~

P 表示每个原子在该样本的 K_b 个有效组之间的软分配。
Pbar 用于在每个组内对原子取加权平均，不是再次沿原子轴做 softmax，
也不强制各组质量相等。返回 mass 方便后续监测组质量和路由退化。

未启用的模板从该样本获得零任务梯度，但优化器动量或权重衰减仍可能改变其数值。
K_b=1 时 softmax 恒为 1，路由分配本身对分数的梯度为零，这是单组情形的数学结果。
有效模板的梯度仍受竞争分数和最终任务损失影响，不保证各组被均匀使用。

## padding、几何不可用与精度

- padding 输入在 MLP 前清零，输出 P/Pbar 的对应原子列为零。
- 全 padding 样本：P、Pbar、mass 全零；前向和反向保持有限。
- geometry_ok=False：仅将 rho 清零并保留 g_ok=0；
  H、c、t 仍参与路由，K_b 仍由原子数决定。
- “合法几何但没有邻居”为 rho=0、g_ok=1，与不可用状态区分。
- softmax 前屏蔽无效组；padding 原子的整行 logits 设为有限值，随后权重清零，
  避免全负无穷行造成 NaN。有效原子没有任何有效组的非法输入会明确报错。
- normalize_assignments 是通用归一化函数：单独调用而未传 group_mask 时，
  使用所有传入的组；SoftRouter 显式传入 AtomCountGroups 生成的 mask。
- 时间编码、这段较小的 MLP、点积、softmax 与组质量计算均至少使用 FP32；
  双精度输入/参数支持 FP64，外层 AMP 不会把这些运算降成半精度。
  参数或输入的转换保留梯度。
- 极小质量组的 Pbar 沿原子轴的和可能明显小于 1；softmax 下溢的零质量组返回零，
  不将其强行变成均匀组。

## 与已有模块连接

~~~python
# 以下实例应在上层模型 __init__ 中注册
condition_encoder = ConditionEncoder()
geometry = PeriodicGeometry()
router = SoftRouter()

# 同一次前向
c_cond = condition_encoder(feats)                # (B,N,128)
geo = geometry(x_t, l_t, feats["atom_pad_mask"])  # 当前状态默认 nm/rad
routes = router(
    H, c_cond, geo.rho, t, geo.geometry_ok, feats["atom_pad_mask"]
)
~~~

上例省略 ConditionEncoder、PeriodicGeometry 的 import。
H 仍由原子主干提供，Router 不负责产生 H 或预测坐标。
此处为原独立测试批次，验证三模块接口相容及梯度贯通；新增主干集成测试见 G1 总文档。

## 默认配置

~~~yaml
_target_: src.models.g1.routing.SoftRouter
h_dim: 256
c_dim: 128
rho_dim: 11
time_dim: 32
hidden_dim: 128
key_dim: 32
max_groups: 32
time_frequency_divisor: 4.0
pool_eps: 1.0e-8
~~~

## 验证

~~~bash
python -m pytest tests/test_soft_router.py -q
python -m pytest tests -q
~~~

2026-10-15，本地 PyTorch 2.6.0+cpu 验证：Router 相关 45 项测试通过；
全仓库 112 项测试通过（112 passed in 27.09s）。

验证范围：

- 固定时间频率、428 维输入顺序、MLP + LayerNorm、缩放点积。
- N=1/10/20/32/40 对应的有效组数；有效模板任务梯度、尾部零任务梯度、单组情形。
- 混合长度批次、非连续 atom_mask、全 padding、无效组与大值/NaN logits。
- 组 mask 在 softmax 前生效，较大样本不会增加较小样本启用的模板数。
- 有效前缀内的模板置换、原子置换、样本独立性及 P/Pbar 归一化。
- 小质量组、极端有限 logits、CPU autocast、半精度输入及参数。
- H/c/rho/t 的有限差分梯度；与 ConditionEncoder、PeriodicGeometry 的实际连接。
- 组数变化期间参数对象及优化器状态保持；独立桥参数和配置实例化。

这些检查不保证真实训练中模板训练充分或分组质量；尚未运行 GPU / 8 GPU 训练，
Router 已接入 Flow，默认关闭；当前尚无真实 G1 训练效果指标。

六模块统一封装与调用方式见 [G1Module](../README.md)。


当前主干接入、开关、checkpoint 兼容性与 263 项全仓库测试记录见 [G1 总文档第 8 节](../README.md#8-atommof-主干接入与开关)。
