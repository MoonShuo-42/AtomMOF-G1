# G1 周期几何模块

根据研究方案第 4.1 节和“E / ρ”架构图实现的独立、确定性几何通路。
从当前带噪状态计算，不读取参考构件、元素、键、构件归属、时间或目标结构。
没有可训练参数，但有效分支对当前坐标及晶胞保留自动微分。

## 目录与职责

| 文件 / 对象 | 职责 | 典型修改 |
| --- | --- | --- |
| lattice.py | 六参数转行晶格；有限性、合法角度、非退化检查 | 更改晶胞接口或数值阈值 |
| images.py / PeriodicImageSearch | 由逆晶格计算安全整数候选范围；限制候选数 | 更换周期枚举实现 |
| moments.py / PeriodicPairMoments | 平滑截断、六个径向基、方向一阶矩、二阶矩、强度 D | 修改几何核 |
| moments.py / scalarize_geometry | 将任意 16 维几何摘要标量化为 11 维 | 复用于未来 U / Z；修改几何不变量 |
| encoder.py / PeriodicGeometry | 批量接口、输入单位、分块、回退、E 与 ρ 的组织 | 修改调度与内存策略 |
| encoder.py / GeometryOutput | E、ρ、几何可用标志的固定输出接口 | 接入上层模型 |
| __init__.py | 公共接口导出 | 对外接口整理 |

配置：configs/g1/geometry/periodic.yaml。
测试：tests/test_periodic_geometry.py。

各部分按职责拆分；当前由 PeriodicGeometry 构造对应对象，未实现任意模块的配置注入。
与 src/models/g1/conditioning 分开：ConditionEncoder 处理固定化学/参考条件，
本模块处理每次调用的当前坐标和晶胞，已通过 G1Bridges 接入 Flow；默认在同次前向的四个桥间共享结果。

## 基本用法

~~~python
from src.models.g1.geometry import PeriodicGeometry

geometry = PeriodicGeometry()  # 注册为上层模型的子模块
result = geometry(x_t, l_t, atom_mask)

E = result.E                       # (B, N, N, 16)
rho = result.rho                   # (B, N, 11)
geometry_ok = result.geometry_ok   # (B,), bool
# 后续路由若需要逐原子可用标量，可展开：
g_ok = geometry_ok[:, None, None].expand(-1, x_t.shape[1], 1)
~~~

| 输入 | 形状 | 默认约定 |
| --- | --- | --- |
| x_t | (B,N,3) | 当前笛卡尔坐标，nm |
| l_t | (B,6) | 当前 (a,b,c,alpha,beta,gamma)，长度 nm、角度 rad |
| atom_mask | (B,N) | True/1 为有效原子 |

默认单位对应原 AtomMOF FlowModule 的内部输入。
特别注意：ConditionEncoder 的 ref_pos 仍是原版 Å 单位，两条输入通路不能混淆。
输入单位必须显式指定，不通过数值大小猜测：

~~~python
# 如果在原 wrapper 的外部调用，使用 Å/degree：
geometry_external = PeriodicGeometry(length_unit="angstrom", angle_unit="degree")
result = geometry_external(cart_coords_angstrom, lattice_angstrom_degree, atom_mask)
~~~

cutoff、min_cell_singular_value 始终以 nm 配置，与输入单位选项无关。

也支持显式晶格矩阵：

~~~python
result = geometry.from_cell(x_t, row_cell, atom_mask)  # row_cell: (B,3,3)
~~~

矩阵每一行是一条晶格向量，x = fractional @ row_cell。
该入口适用于已有矩阵或共同旋转测试；矩阵和坐标必须处于同一坐标系。
若旋转 O 作用于列向量，行数组应同时变为 x_t @ O.T 和 row_cell @ O.T。
仅旋转坐标、再由不变的六参数重建规范晶胞，不是同一几何变换。

## 计算定义

### 晶胞与周期覆盖

六参数按原基座“c 轴沿 z”的规范朝向转换，返回行晶格 L：

~~~text
a_vec = (a sin(beta), 0, a cos(beta))
eta = (cos(gamma) - cos(alpha) cos(beta)) / sin(beta)
b_vec = b * (eta, sqrt(sin(alpha)^2 - eta^2), cos(alpha))
c_vec = (0, 0, c)
L = stack(a_vec, b_vec, c_vec)
~~~

对原子对 i、j：

~~~text
delta_f = (x_j - x_i) @ inverse(L)
b_u = cutoff * norm(column_u(inverse(L)))

ceil(-delta_f_u - b_u - margin) <= n_u
n_u <= floor(-delta_f_u + b_u + margin)

d_ij(n) = x_j - x_i + n @ L
~~~

以 float64 搜索安全包围盒，向外扩少量数值余量，再按真实三维半径过滤。
汇总半径内全部覆盖的平移副本，不使用固定 27 像的最近镜像替代。
只排除 i=j 且 n=0 的自身边；保留非零平移的自身副本。
不同原子重合时仍保留该边，不能用“距离为零”作为统一删除条件。
这里的边表示当前几何接近，不是化学键，也不限制同构件。

### E 的 16 个通道

默认 cutoff = r0 = 0.6 nm，delta = 0.1，sigma = 0.2：

~~~text
y_n = d_n / r0
q_n = ||y_n||^2
w_n = max(1 - q_n, 0)^3
D = sum_n w_n
beta_n = w_n / (D + delta)

mu_m = m / 5, m = 0,...,5
phi_m(q) = exp(-(q - mu_m)^2 / (2 sigma^2))
r_m = sum_n beta_n * phi_m(q_n)
v = sum_n beta_n * y_n
Q = sum_n beta_n * y_n y_n^T

E_ij = [r_0,...,r_5, vx,vy,vz, Qxx,Qyy,Qzz,Qxy,Qxz,Qyz, D]
~~~

所有输出通道无量纲。径向基中心位于归一化平方距离 q 轴；
Q 是原始二阶矩，不是中心化协方差。
delta 是有意义的衰减尺度，不替换为极小 epsilon：
单像接近半径边界时，其贡献应逐渐消失。

### 先汇总，再计算 ρ

~~~text
G_i = sum_{j valid} E_ij
chi([r,v,Q,D]) = [r_0,...,r_5, D, ||v||^2, tr(Q), tr(Q^2), v^T Q v]
rho_i = chi(G_i)
~~~

ρ 不除以全结构原子数 N。必须先对 j 求和，再调用 chi；
它通常不等于 sum_j chi(E_ij)。
tr(Q^2) 包含两倍的非对角分量平方，不能直接对六维压缩向量求平方和。

E 中 v、Q 在共同旋转下分别按向量、二阶张量变换；ρ 在共同旋转下不变。
这不代表整个 G1 网络严格旋转等变，也不保证这些低阶统计可以区分手性。

## 数值策略与异常语义

- 默认每对最多 125 个候选整数平移。此上限作用于安全包围盒，
  不是球内最终副本数；超限使整个样本的几何通路不可用，绝不静默截断。
- 非有限有效原子坐标、非法六参数、奇异/病态晶胞和无法可靠表示的整数范围，
  同样返回 geometry_ok=False，并将该样本的 E、ρ 全部置零。
- 合法晶胞但半径内无邻居：E、ρ 可以全零，而 geometry_ok=True。
  全 padding 样本返回 geometry_ok=False。
- padding 行和列均清零；padding 坐标即使为 NaN/Inf 也不参与几何或有效性判断。
- 回退只禁用新几何分支，不在此模块中删除样本。接入后上层必须继续保留
  原子/条件通路及训练、评估分母，并记录 geometry_ok 的比例。
- 晶胞数值保护默认最小奇异值 1e-6 nm、最大条件数 1e6。
  六参数转换还检查角度范围、sin(beta) 和平方根项（1e-12 级阈值）。
  这些是数值保护，不是材料物理阈值。
- 禁用内部 autocast；FP16/BF16 输入提升为 FP32，FP64 输入保留 FP64。
  晶胞验证和离散搜索使用 FP64。输出为 FP32 或 FP64。
- 离散候选和可用性判断不求导；位移、核、矩和 ρ 使用张量计算保留梯度。
  超限/合法性切换仍可能不连续，不承诺全局处处光滑。

## 内存、复用与配置

默认按全批最多 4096 个有效原子对分块，不一次展开全部 BN²×125 位移。
先在 planning.py 中批量求逆、计算原子对边界、验证整样本覆盖，再压紧全部有效原子对。
各块复用已算边界；候选槽宽度采用本批有效候选数的最大值，不再固定填满 125 个槽。
离散搜索仍不求导，活跃位移和矩保持梯度。逐样本 Python 循环和标量同步已移除；
批级 nonzero 和最大候选数读取仍有动态同步，不能声称完全无同步。

activation_checkpointing 默认 false，与本次 small 训练配置对齐。
显式开启时，仅在输入需要梯度的情况下按块重算，用额外计算换取候选激活内存。

最终仍返回 dense E，内存为 O(BN²×16)。
N=1356 时，单份 FP32 E 约 112 MiB，尚不含其他张量及反向内存。
批量搜索边界同样有 O(BN²) 临时存储；分块限制的是候选镜像展开，不消除 N² 存储。
几何定义不变：超限样本整条回退；合法无邻居样本仍返回 geometry_ok=True。

同次网络调用的多座桥可共享一次计算的 E、ρ；外层更新 x_t/l_t 后需重新计算。
模块不维护隐式缓存，不与 ConditionEncoder 的参考条件混用。

~~~yaml
_target_: src.models.g1.geometry.PeriodicGeometry
cutoff: 0.6
radial_sigma: 0.2
delta: 0.1
max_images: 125
pair_chunk_size: 4096
bounds_margin: 1.0e-9
min_cell_singular_value: 1.0e-6
max_cell_condition: 1.0e6
length_unit: nm
angle_unit: radian
activation_checkpointing: false
~~~

## 验证

~~~bash
python -m pytest tests/test_periodic_geometry.py -q
python -m pytest tests -q
~~~

2026-10-15 本地 PyTorch 2.6.0+cpu 验证：新增几何测试 36 项通过；
完整仓库测试 67 项通过（67 passed in 23.04s）。

测试覆盖方案中的跨边界数值算例、斜晶胞大范围暴力枚举、原版晶胞朝向、
自身周期副本、原子碰撞、共同旋转、周期代表元变换、置换、padding、
无邻居与不可用标志、候选上限、平滑截断、坐标和六参数有限差分梯度、
分块/checkpoint 一致性、混合精度保护及显式单位转换。

原版晶胞 helper 的对照测试直接提取并执行其源码函数，避免导入其所在文件中
与该纯 Torch 函数无关的结构分析依赖；不复制公式作为“原版”替身。

本地验证仅覆盖 CPU 数值与自动微分；GPU 性能、真实数据可用率、8 GPU 集成、
训练稳定性和模型效果仍需后续实测。

六模块统一封装与调用方式见 [G1Module](../README.md)。


主干接入见第 8 节；最新批量几何优化与验证见 [G1 总文档第 10 节](../README.md#10-共享条件批量几何与关闭重算2026-10-16)。
