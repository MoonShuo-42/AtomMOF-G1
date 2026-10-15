# G1 六模块统一封装

`G1Module` 将已经实现的六个模块组合为一次完整的原子特征更新。
封装层负责连接、接口检查与结果组织，计算仍由各子模块完成。
默认输入 H 为 (B,N,256)，返回 H_updated 为 (B,N,256)。

## 1. 组成与数据流

| 注册名称 | 子模块 | 职责与默认输出 |
| --- | --- | --- |
| condition_encoder | ConditionEncoder | 静态条件 c_cond：(B,N,128) |
| geometry | PeriodicGeometry | 当前周期几何 E：(B,N,N,16)、rho：(B,N,11)、geometry_ok |
| router | SoftRouter | 动态分组 P/Pbar：(B,K,N)、group_mask |
| aggregation | GeometryAggregator | U：(B,N,K,16)、Z：(B,K,K,16) |
| group_interaction | GroupInteraction | S 与 S_plus：(B,K,128) |
| readback | AtomReadback | readback：(B,N,128)、delta_H 与 H_updated：(B,N,256) |

~~~text
feats ── ConditionEncoder ── c_cond ────────────┐
x_t,l_t,mask ── PeriodicGeometry ── rho,g_ok ────┼─ SoftRouter(H,c,rho,t,g_ok,mask)
                                    E ──┐     │           │
                                       │                 P / Pbar / group_mask
                                       └─ GeometryAggregator(E,Pbar) ── U,Z
H,Pbar,Z ── GroupInteraction ── S_plus
H,S_plus,U,Z,P ── AtomReadback ── readback,delta_H,H_updated
~~~

连接规则：

- 从 feats 的 atom_pad_mask 取得唯一原子 mask，贯穿所有分支。
- Router 每次针对当前 H、t 重新计算；沿用 K_b=min(32,N_b)，全空批次保留一个被屏蔽存储槽。
- 同一份 Pbar 同时用于 U/Z 和 S 的汇总；同一路由的 P 用于 readback 的 log 先验。
- geometry_ok 贯穿 Router、几何汇总、组注意力和读回；几何不可用时内容路径继续工作。
- AtomReadback 已执行 H + delta_H，封装层不会重复加残差。
- 新封装自身不增加可学习投影或模板，不改变六个子模块内部公式。

这相当于一个可复用的 G1 桥接模块，不包含原子主干或坐标/晶格速度预测头。
H 由外部主干提供。新增 G1Bridges 将独立 G1 默认插入主干 DiT 第 6/12/18/24 层之后，
通过默认关闭的配置开关控制；具体接入与训练/推理方式见第 8 节。

## 2. 本地文件与模块化方式

| 文件 | 职责 |
| --- | --- |
| module.py | G1Module，构造子模块并串联六阶段 |
| bridges.py | G1Bridges，四处独立桥、同次前向几何复用与激活重算 |
| outputs.py | G1Context、G1Intermediates、G1Output 数据接口 |
| validation.py | 子模块维度、pool_eps 和上下文形状检查 |
| __init__.py | 对外导出 |
| configs/g1/default.yaml（仓库根目录下） | 组合 configs/g1/ 下的六份子模块配置 |
| tests/test_g1_module.py（仓库根目录下） | 封装与端到端测试 |

六个子模块目录统一放在 src/models/g1/ 下，各自保持独立：
[conditioning](conditioning/README.md)、[geometry](geometry/README.md)、
[routing](routing/README.md)、[aggregation](aggregation/README.md)、
[group_attention](group_attention/README.md)、[readback](readback/README.md)。

当前代码目录：

~~~text
src/models/g1/
├── __init__.py
├── module.py
├── bridges.py
├── outputs.py
├── validation.py
├── README.md
├── conditioning/
├── geometry/
├── routing/
├── aggregation/
├── group_attention/
└── readback/
~~~

统一入口仍为 from src.models.g1 import G1Module；
独立子模块改用 from src.models.g1.conditioning import ConditionEncoder 等路径。
六个子模块的配置也统一放入 configs/g1/，测试仍放在 tests/；配置加载路径、Hydra defaults 和测试已同步更新。
本次为目录整理，子模块注册名称和 state_dict 键不变。

配置目录与代码目录分别集中管理：

~~~text
configs/g1/
├── default.yaml
├── bridges.yaml
├── condition_encoder/static.yaml
├── geometry/periodic.yaml
├── router/adaptive.yaml
├── aggregation/geometry.yaml
├── group_attention/default.yaml
└── readback/default.yaml
~~~

原 configs/ 下的六个平级配置目录已移入此处。default.yaml 使用相对于 g1/ 的配置组路径；
对外仍通过 compose(config_name="g1/default") 加载，实例化入口仍为 cfg.g1，
参数覆盖仍使用 g1.router.max_groups 等键。group_attention 配置组装入 group_interaction 字段。
独立子模块可继续 OmegaConf.load 对应 YAML，例如 configs/g1/router/adaptive.yaml；
统一入口含有 defaults，必须通过 Hydra compose 展开。

构造函数可以直接传入任一子模块实例，便于更换实现、参数设置或复用已加载的权重。
未传入的组件采用默认实现；已传入的实例原样注册，不复制或重新初始化其参数。
不同的 G1Module() 默认拥有各自独立参数；显式复用同一实例则共享该实例的参数。
所有组件都通过 nn.Module 注册，支持 .to(device)、.train()/.eval()、state_dict 和优化器参数收集。
无可学习参数的几何模块仍包含需要保存的 radial_centers buffer。

## 3. 一次调用六个模块

在上层模型的 __init__ 中创建一次并注册 g1，不要在每次 forward 时重新创建。

~~~python
from src.models.g1 import G1Module

g1 = G1Module().to(device)

# forward 中：所有输入、条件字段与模型位于同一设备
result = g1(
    H=H,                       # (B,N,256)，外部原子主干特征
    x_t=x_t,                   # (B,N,3)，当前笛卡尔坐标，默认 nm
    l_t=l_t,                   # (B,6)，默认 nm / radian
    t=t,                       # (B,) 或 (B,1)
    feats=feats,                # 静态条件字段，含 atom_pad_mask
)
H_next = result.H_updated       # (B,N,256)
delta_H = result.delta_H        # (B,N,256)
readback = result.readback      # (B,N,128)
group_counts = result.group_counts  # (B,)
geometry_ok = result.geometry_ok    # (B,)，bool
assert result.intermediates is None
~~~

返回值是 G1Output，并非单个 Tensor；传给后续主干时使用 result.H_updated。

feats 沿用原有 ConditionEncoder 的接口：

| 字段 | 形状 | 约定 |
| --- | --- | --- |
| ref_pos | (B,N,3) | 构件参考坐标，沿用原始 Å 单位 |
| ref_charge、ref_element | (B,N) | 电荷和元素 |
| block_type、block_index | (B,N) | 构件类型与归属编号 |
| atom_pad_mask | (B,N) | bool 或 0/1；True/1 为有效原子 |
| bond_types | (B,N,N) | 稠密键类型整数矩阵 |

参考条件 ref_pos 与当前状态 x_t 用途和默认单位不同，封装层不额外混合或转换单位。
l_t 顺序为 [a,b,c,alpha,beta,gamma]，单位由 geometry 的配置控制。
显式传入行晶格矩阵 (B,3,3) 时设置 lattice_format="cell"，避免根据形状猜测：

~~~python
result = g1(H, x_t, row_cell, t, feats, lattice_format="cell")
~~~

ConditionEncoder 仍要求参考条件采用原 collator 的有限、零填充约定，
分类编号必须在有效范围内。不要将下游几何分支可忽略 NaN padding 的行为误用于参考条件。
H、当前坐标等动态输入的 padding 按已有子模块规则屏蔽。

## 4. 查看中间结果

~~~python
result = g1(H, x_t, l_t, t, feats, return_intermediates=True)
detail = result.intermediates

c_cond = detail.context.c_cond
E, rho, g_ok = detail.context.geometry
P, Pbar = detail.routing.P, detail.routing.Pbar
group_mask = detail.routing.group_mask
U, Z = detail.relations.U, detail.relations.Z
S, S_plus = detail.groups.S, detail.groups.S_plus
group_bias = detail.groups.geometry_bias
readback_bias = detail.atom_readback.geometry_bias
readback_weights = detail.atom_readback.attention_weights
~~~

中间结果保留梯度，不执行 detach 或复制。便于添加诊断和辅助损失。
默认不在返回对象中保留这些额外引用，但训练反向仍会按需要保存激活，
并不等于避免构造稠密 E 或 value。日志中长期保存中间结果前需自行 detach / 转到 CPU。

## 5. 显式复用条件和几何

普通 forward 每次计算所有六个模块，没有隐式缓存。
为后续多个桥接位置或采样复用提供三个独立入口：

- encode_condition(feats)：只编码静态条件。
- prepare(feats,x_t,l_t,c_cond=None)：准备条件和当前几何，返回 G1Context；提供 c_cond 时跳过条件编码。
- forward_with_context(H,t,context)：复用条件/几何，但重新计算路由、U/Z、组交互和读回。

同一状态、同一原子排列下更新不同的 H：

~~~python
context = g1.prepare(feats, x_t, l_t)
result1 = g1.forward_with_context(H1, t, context)
result2 = g1.forward_with_context(H2, t, context)
~~~

推理时静态条件可以跨采样步复用，但每步当前坐标和晶格变化后必须重算几何：

~~~python
import torch

g1.eval()
with torch.no_grad():
    c_cond = g1.encode_condition(feats)
    # 每个采样步都使用该步新的 x_t / l_t
    context = g1.prepare(feats, x_t, l_t, c_cond=c_cond)
    result = g1.forward_with_context(H, t, context)
~~~

缓存有效性由调用者保证。条件、原子顺序、mask、坐标、晶格或几何配置改变后，
旧 context 不能直接复用；同形状无法证明是同一结构，接口检查不会推断来源。
训练时 context 不自动 detach；只在当前计算图内复用，不能跨优化器更新保留旧条件或旧图。
若外部提供的 c_cond 与本实例的编码器无关，本实例编码器就不参与该路径的梯度；
多卡训练的参数参与与缓存生命周期需要上层模型统一管理。

## 6. 子模块替换与配置

~~~python
from src.models.g1.conditioning import ConditionEncoder
from src.models.g1.routing import SoftRouter
from src.models.g1 import G1Module

g1 = G1Module(
    condition_encoder=ConditionEncoder(depth=4),
    router=SoftRouter(max_groups=32),
)
~~~

未提供 router 时，其 c_dim 从 condition_encoder.output_dim 取得；
未提供 group_interaction 时，其 h_dim 从 router 取得；
未提供 readback 时，其 H/组宽度及 pool_eps 从上游取得，其他参数沿用默认值。
显式提供的子模块不会被封装层擅自改写，以下不一致会在构造时直接报错：
条件输出宽度、rho 的 11 通道、H 宽度、S_plus 宽度、Router/readback 的 pool_eps。
自定义替代实现需保持调用接口及 validation.py 中使用的维度属性。

configs/g1/default.yaml 通过相对 Hydra defaults 组合该目录下的六份子配置，不复制各自的默认参数。
因此应使用 Hydra compose，再 instantiate；单独 OmegaConf.load 不会展开 defaults。

~~~python
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

# 从仓库根目录运行；如果外部应用已经初始化 Hydra，沿用其配置上下文。
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
    cfg = compose(
        config_name="g1/default",
        overrides=[
            "g1.condition_encoder.depth=3",
            "g1.router.max_groups=32",
            "g1.readback.value_hidden_dim=128",
        ],
    )
g1 = instantiate(cfg.g1)
~~~

组合配置显式构造所有六个组件，修改关联维度或 pool_eps 时须同步相应子配置。
default.yaml 保留独立实例化入口。新增 bridges.yaml 复用它并由 configs/model/simple.yaml 组合，
接入训练构造与 checkpoint 保存；推理根据 checkpoint 还原架构。

## 7. 验证与接入范围

~~~bash
python -m pytest tests/test_g1_module.py -q
python -m pytest tests -q
~~~

2026-10-15，本地 PyTorch 2.6.0+cpu：统一封装测试共 25 项；六个子模块迁入 g1 后，全仓库 240 项测试通过（240 passed in 38.32s），包括独立子模块、Hydra 配置组合与完整链路验证。代码格式检查通过。

同日进一步将六个配置目录迁入 configs/g1/，更新 defaults 和配置加载路径后，
重新运行全仓库测试：240 passed in 34.99s。独立配置加载、Hydra 组合、参数覆盖和完整链路均通过。

检查包括：六个组件注册与同一份 P/Pbar 的传递、与手动六阶段串联的数值和参数梯度对齐、
残差只执行一次、显式上下文复用与失效边界、静态条件复用、动态组数和样本独立性、
几何不可用时保留内容路径、全 padding 的有限零梯度、原子置换、行晶格接口、
CPU autocast、参数独立性与状态保存恢复、Hydra 配置组合和子模块参数覆盖。

已完成独立 G1、四桥调度及 Flow 训练/推理接入，默认关闭；尚未进行 G1 GPU 训练。
本地数值测试不代表实际训练效果或显存性能。


## 8. AtomMOF 主干接入与开关

G1Bridges 在 FlowModule 中注册四个独立 G1Module。TokenTransformer 将回调传入主干 DiT，
每层完成后按一基编号检查插入位置，默认依次在 6、12、18、24 层之后更新隐藏特征。
原始原子编码器、原子解码器、pair 特征和时间调制的调用方式保留。
目前主干 token 与原子仍一一对应；若未来真正执行 atom-to-token 分组，需要另行适配 G1 的原子轴。

桥的 H 宽度取实际 model.token_s，可能为 256 或 512，不硬编码为 256。
bridges.yaml 将 router、group_interaction、readback 的 h_dim 联动到 model.token_s；
组特征与条件输出的默认 128 维保持原设定。四个桥的条件编码器、Router、组交互和读回参数均独立。

配置入口为 configs/g1/bridges.yaml，位于 model.flow_model_args.g1：

~~~yaml
model:
  flow_model_args:
    g1:
      enabled: false
      insert_after: [6, 12, 18, 24]
      activation_checkpointing: false
      share_geometry: true
      module: # 由 default.yaml 组合出的 G1Module 配置
        ...
~~~

配置中的 builtins.dict 与 _recursive_: false 用于延迟子模块实例化。
关闭时既不构造 G1 权重，也不调用条件编码、几何或多桥回调；原版 state_dict 键不增加。
插入层必须递增、唯一，且不超过当前主干深度，错误配置会在构造时直接报错。

在现有训练命令中追加下列参数即可启用，其余数据路径、Small 尺寸和阶段配置沿用原命令：

~~~bash
model.flow_model_args.g1.enabled=true
~~~

省略或设为 false 使用 baseline。其他可覆盖项：

~~~bash
model.flow_model_args.g1.insert_after='[6,12,18,24]'
model.flow_model_args.g1.activation_checkpointing=true
model.flow_model_args.g1.module.router.max_groups=32
~~~

训练 forward、验证采样、预测采样和 refine_final 均经过同一 FlowModule。
主干激活重算和 G1 激活重算分别受原 activation_checkpointing 与新桥开关控制，均采用非重入重算。
训练中的架构由构造配置确定；不支持在 DDP 已注册参数后通过运行时开关改变参与训练的参数集合。

### 8.1 上下文与多样本

同一 Flow forward 内，默认只计算一次当前 x_t/l_t 的无参数几何，四个桥共用 E/rho/g_ok。
各桥仍分别运行自己的 ConditionEncoder，在原始 batch 上编码后按 repeat_interleave 扩展 multiplicity。
每个插入位置用当前隐藏特征重新计算自己的路由、U/Z、S_plus 和读回；不重复添加残差。
上下文只保存在本次调用的局部变量中，不跨采样步缓存、不 detach，也不跨优化器更新保留计算图。
share_geometry=false 可关闭几何复用；共享模式要求各桥采用同一配置的无参数 PeriodicGeometry。
Flow 已将当前坐标/晶格转换为 nm/radian，集成模式会校验该单位约定。

### 8.2 Checkpoint 与预测开关

| 场景 | 行为 |
| --- | --- |
| 原 baseline checkpoint | 缺少 g1 配置时按关闭处理，严格加载原有权重 |
| G1 checkpoint | 保存四桥参数与 flow_model_args.g1，推理时自动恢复 |
| baseline 权重初始化 G1 训练 | 启用 G1，并使用 pretrained_ckpt_path；主干尺寸应一致，新桥随机初始化 |
| 继续 G1 训练 | 同样启用 G1、保留相同结构，使用 resume_ckpt_path 恢复优化器与训练进度 |

预测结构以 checkpoint 为准；命令中的 model 配置不覆盖其架构。
预测时使用 predict_args.g1_enabled：

~~~bash
# 默认：按 checkpoint 的 G1 状态运行
predict_args.g1_enabled=null

# 加载 G1 checkpoint 后跳过桥，进行消融
predict_args.g1_enabled=false

# 要求 checkpoint 含 G1 权重；baseline checkpoint 会报错
predict_args.g1_enabled=true
~~~

显式覆盖时预测目录名附加 _g1-False 或 _g1-True，避免覆盖默认预测。
默认 null 保留原输出目录命名。关闭 G1 checkpoint 中的桥不会把已训练主干还原为官方 baseline 权重。
程序调用可在 model.eval() 后使用 model.structure_module.flow_model.set_g1_enabled(False)。
构造时关闭的模型不能在运行时凭空开启 G1；需要包含 G1 权重的模型。

### 8.3 接入验证（2026-10-15）

新增 tests/test_g1_integration.py：23 项集成测试通过。
提交前复核：全仓库 263 passed, 2 warnings in 80.78s；两条警告来自 CPU 微型训练测试的无验证集与少量 dataloader workers。
覆盖四个插入位置、独立参数、multiplicity、当前几何复用与失效、主干/桥激活重算的梯度一致性、
真实训练损失与两步 Lightning 优化、ODE 与最终 refinement、严格 checkpoint 恢复、baseline 权重初始化 G1、
训练/预测 Hydra 入口及预测开关。两进程 Gloo DDP 使用 find_unused_parameters=False 连续训练两步通过，
其中一个 rank 的样本进入几何回退分支。

另与修改前保存的 Flow/TokenTransformer/DiT 源码进行独立比较：关闭 G1 时，
初始化 RNG、481 个状态项、前向输出、全部参数梯度与输入梯度逐项完全相等，
主干激活重算关闭和开启两种情况均通过（小尺寸合成输入、CPU）。

本地环境为 PyTorch 2.6.0+cpu；测试仅隔离未使用的可选 MLIP 后端，主干、六模块、采样器和 Lightning 均为真实实现。
没有运行服务器 8 GPU、MLIP steering、torch.compile 性能或完整数据集训练；尚无 G1 生成效果指标。
