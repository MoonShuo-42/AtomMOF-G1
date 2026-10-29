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
H 由外部主干提供。G1Bridges 默认在主干 DiT 第 6/12/18/24 层之后执行更新，
通过默认关闭的配置开关控制；具体接入与训练/推理方式见第 8 节。

### 与 AtomMOF 主干的边界

G1 仍是可选的模块扩展。六个子模块及多桥调度位于 `src/models/g1/`，
主干 `FlowModule` 只负责按开关构造桥、准备上下文并传入层后回调。
DiT 的 `after_layer` 接口不包含路由、周期几何、U/Z 汇总或读回公式。

- `model.flow_model_args.g1.enabled=false`：不导入或构造 G1，不注册 G1 参数，
  不执行条件编码、几何搜索或桥更新；训练/推理入口同时关闭 G1 专用 CPU 布局准备。
- `enabled=true`：主干在配置的层后调用 G1。四桥共享的是 G1 自己的
  ConditionEncoder；它与原始 AtomMOF 的原子编码器不共享可学习参数。
- G1 条件编码复用主干的 Attention/MLP 类和公共工具，这是代码复用。
  公共执行策略由各组件声明其所需开关，不反向导入 G1 类。
- 对已加载的 G1 checkpoint 使用 `predict_args.g1_enabled=false` 是推理消融：
  保留 checkpoint 参数但跳过 G1 计算和专用数据准备；不能将其当成重新训练的 baseline。

训练和推理入口在模型构造或加载完成后调用 `configure_data_execution`，
以实际模型状态设置数据预处理。推理以 checkpoint、显式消融开关和运行时 execution
配置为准，无需另行指定 `data.prepare_g1_metadata=true` 来修正默认训练配置的差异。
自行调用 DataModule/Trainer 的脚本应在创建 DataLoader 前调用同一接口；
没有预处理元数据的直接模型调用仍可使用设备端回退路径。

`execution=math/optimized` 是可同时用于 baseline 和 G1 的实现优化开关，
与是否加入 G1 相互独立。因此关闭 G1 表示使用 baseline 架构，不表示还原上游仓库
每一行实现；参数、输入/输出接口与计算精度的对齐由回归测试检查。
本次边界调整不改变模型公式、参数键/形状、训练样本、batch 预算或精度设置。

隔离验证（2026-10-20，基于 `375d721` 的本地修改）：`python -m pytest tests -q`
共 **434 项通过、4 条测试环境警告**。包括禁止导入整个 G1 包后的 baseline
前向/反向、训练/推理入口的数据准备、checkpoint 开关与运行时配置不一致、
六个子模块、编译开关重置、采样及双进程 Gloo DDP。
另将修改前后的执行策略用于同权重的小模型，baseline/G1 两条路径的 CPU 输出和
参数梯度逐项完全一致。small 参数量仍为 baseline 38,376,320、G1 40,525,216。
这些是本地 CPU 正确性验证，不是本次改动的服务器加速或显存测量。

## 2. 本地文件与模块化方式

| 文件 | 职责 |
| --- | --- |
| module.py | G1Module，构造子模块并串联六阶段 |
| bridges.py | G1Bridges，共享条件编码、四处独立更新、同次前向几何复用与可选重算 |
| outputs.py | G1Context、G1Intermediates、G1Output 数据接口 |
| validation.py | 子模块维度、pool_eps 和上下文形状检查 |
| __init__.py | 对外导出 |
| configs/g1/default.yaml（仓库根目录下） | 组合 configs/g1/ 下的六份子模块配置 |
| tests/test_g1_module.py（仓库根目录下） | 封装与端到端测试 |
| tests/test_g1_isolation.py（仓库根目录下） | baseline 无 G1 导入运行、数据准备与训练入口隔离验证 |

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

default.yaml 使用相对于 g1/ 的配置组路径；
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
            "g1.condition_encoder.depth=2",
            "g1.router.max_groups=32",
            "g1.readback.value_hidden_dim=128",
        ],
    )
g1 = instantiate(cfg.g1)
~~~

组合配置显式构造所有六个组件，修改关联维度或 pool_eps 时须同步相应子配置。
default.yaml 保留独立实例化入口。新增 bridges.yaml 复用它并由 configs/model/simple.yaml 组合，
接入训练构造与 checkpoint 保存；推理根据当前方案的 checkpoint 还原架构。

## 7. 验证与实验状态

~~~bash
python -m pytest tests -q
~~~

本地回归覆盖六模块前向/梯度、紧凑条件与数学参考对照、四桥共享条件及梯度累加、
动态组数与 mask、周期几何、当前 checkpoint 保存恢复、baseline 权重初始化 G1、
真实 Flow 损失与采样、两进程 Gloo DDP。

2026-10-16 本次清理验证（PyTorch 2.6.0+cpu）：受影响测试 124 项通过，
其余测试 170 项通过，合计 **294 项通过**。两条警告来自微型 Lightning 测试的
无验证 dataloader 和较少 workers。组件基准脚本也通过 CPU 前向/反向运行检查；
该小规模运行只验证脚本可用，不作为性能测量结论。

服务器已完成八卡 smoke 测试，并开始 G1 small 阶段一正式训练。
2026-10-16 用户反馈优化后训练段约 3 分半；截图为进行中的进度估计，
不是固定实验条件下完成整轮的基准，也不包含完整验证阶段。
这些结果说明代码已在服务器运行；收敛与生成指标仍需后续实验评估。

清理前曾直接对照接入前的 2e21e0e 源码：CPU FP32、24 层且缩小测试宽度，
主干重算开/关 × multiplicity 1/2 共四组，G1 关闭时初始化 RNG、参数、
前向输出、参数梯度和输入梯度完全一致。当前回归继续检查 baseline 关闭路径。

## 8. AtomMOF 主干接入与开关

G1Bridges 在 FlowModule 中注册四个共享条件编码器的 G1Module。TokenTransformer 将回调传入主干 DiT，
每层完成后按一基编号检查插入位置，默认依次在 6、12、18、24 层之后更新隐藏特征。
原始原子编码器、原子解码器、pair 特征和时间调制的调用方式保留。
目前主干 token 与原子仍一一对应；若未来真正执行 atom-to-token 分组，需要另行适配 G1 的原子轴。

桥的 H 宽度取实际 model.token_s，可能为 256 或 512，不硬编码为 256。
bridges.yaml 将 router、group_interaction、readback 的 h_dim 联动到 model.token_s；
组特征与条件输出的默认 128 维保持原设定。始终共享一套条件编码器；Router、组交互和读回参数仍逐桥独立。

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
在原始 batch 上只运行一次共享 ConditionEncoder，再按 repeat_interleave 扩展 multiplicity；
四桥引用同一 c_cond 计算图。梯度不 detach，四个插入位置的梯度累加到共享参数。
每个插入位置用当前隐藏特征重新计算自己的路由、U/Z、S_plus 和读回；不重复添加残差。
上下文只保存在本次调用的局部变量中，不跨采样步缓存、不 detach，也不跨优化器更新保留计算图。
share_geometry=false 可关闭几何复用；共享模式要求各桥采用同一配置的无参数 PeriodicGeometry。
Flow 已将当前坐标/晶格转换为 nm/radian，集成模式会校验该单位约定。

### 8.2 Checkpoint 与预测开关

| 场景 | 行为 |
| --- | --- |
| 原 baseline checkpoint | 缺少 g1 配置时按关闭处理，严格加载原有权重 |
| 当前方案 G1 checkpoint | 保存所有插入位置的参数与 flow_model_args.g1，推理时自动恢复 |
| baseline 权重初始化 G1 训练 | 启用 G1，并使用 pretrained_ckpt_path；主干尺寸应一致，新桥随机初始化 |
| 继续 G1 训练 | 同样启用 G1、保留相同结构，使用 resume_ckpt_path 恢复优化器与训练进度 |
| G1 扩大 K 后进入下一阶段 | 使用 pretrained_ckpt_path 和 pretrained_g1_expand_groups=true，仅扩展路由模板行，其他权重严格加载 |
| G1 增加插入位置后进入下一阶段 | 使用 pretrained_g1_expand_bridges=true，保留旧位置并对新增桥的残差输出投影零初始化，可同时显式扩大 K |

例如从 K 上限 32 的 stage-1 checkpoint 初始化 K 上限 64 的 stage-2，在第二阶段命令中设置：

~~~bash
model.flow_model_args.g1.module.router.max_groups=64 \
pretrained_g1_expand_groups=true \
pretrained_ckpt_path="$PRETRAINED_CKPT" \
resume_ckpt_path=null
~~~

扩容只允许增大 max_groups，不允许缩小、缺失/多余权重、改变其他参数形状或 dtype。
四座桥各自的 template_bank.weight 从 (32,32) 变为 (64,32)：前 32 行完整加载，
后 32 行保留新模型按 seed 构造时的标准正态初始化。其他权重逐值恢复，随后新旧参数共同训练。
small 默认四桥总参数量从 40,525,216 增至 40,529,312，仅新增 4,096 个参数。
此操作不恢复上一阶段的优化器、调度器或 epoch；扩容开关与 resume_ckpt_path 不能同时使用。
K 相同时不需要此开关；K=64 的新 checkpoint 保存新容量，后续同容量推理和断点续训照常加载。
新增组会参与 softmax，保留旧权重不代表扩容后初始输出相同；实际 K 仍为 min(max_groups,N)。

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

只维护当前 G1 架构和配置；除显式 K 扩容和下述插入位置扩展外，不提供旧 G1 架构的自动迁移或恢复模式。
原始 AtomMOF baseline 的加载、初始化 G1 训练和关闭开关保留。
共享条件编码器的四组 state_dict 名称是同一套权重的别名；加载时检查它们一致，
避免损坏或人工修改的 checkpoint 静默覆盖共享权重。

### 8.3 从 Stage-1 增加 G1 插入密度

`pretrained_g1_expand_bridges=true` 只用于 `pretrained_ckpt_path` 权重迁移，
要求 `resume_ckpt_path=null`。旧位置必须是新位置的严格子集，不能删除或移动旧桥；
K 保持不变时不要开启 `pretrained_g1_expand_groups`，增大 K 时同时开启它。
此入口针对原始 G1、CPU 构造及 `ema=false`，与 G1-New 迁移入口互斥。

实现先按旧插入位置构造目标 K 的模型，完整加载旧权重并执行既有 K 扩容，
再在独立的 PyTorch CPU RNG 上下文中构造更密的模型。原有位置新增模板行的初始化、
迁移后的 CPU RNG 状态均与既有 K 扩容流程一致。旧权重和所有共享条件编码器别名完整复制；
新增桥保持内部正常初始化，仅 `readback.residual.output_projection.weight` 置零。
随后组装完整 state_dict 并严格加载，源 checkpoint 不会被修改。

新增桥初始满足 `H_out = H + 0 * readback = H`（有效原子且中间值有限）。
第一次反传时末端投影通常有非零梯度，前面的独立参数任务梯度为零；投影更新后，
内部参数开始获得任务梯度。没有新增门控、归一化、冻结策略或低精度路径。
这保证加桥本身的初始恒等行为；K32→K64 仍会改变路由，不能声称与原始 K32 输出相同。

插入位置、K、训练轮数和保存间隔均由命令行控制，不新增专用实验配置文件。
例如以现有 8 GPU、最多 300 原子、平方预算 3,000,000、B32/累积2、
skip_too_big=false、seed42 的 Stage-2 完整命令为基础，设置下列参数。
数据路径、small 模型尺寸、LR 上限 1e-4 和 warmup 10,000 次优化器更新等沿用原命令。

~~~bash
execution=optimized \
model.flow_model_args.g1.enabled=true \
model.flow_model_args.g1.insert_after='[3,6,9,12,15,18,21,24]' \
model.flow_model_args.g1.module.router.max_groups=64 \
pretrained_g1_expand_bridges=true \
pretrained_g1_expand_groups=true \
pretrained_ckpt_path="$STAGE1_CKPT" \
resume_ckpt_path=null \
trainer.max_epochs=200 \
callbacks.model_checkpoint.monitor=null \
callbacks.model_checkpoint.save_top_k=-1 \
callbacks.model_checkpoint.save_last=true \
callbacks.model_checkpoint.save_weights_only=false \
callbacks.model_checkpoint.every_n_epochs=100 \
callbacks.model_checkpoint.every_n_train_steps=null \
callbacks.model_checkpoint.train_time_interval=null \
callbacks.model_checkpoint.save_on_train_epoch_end=true \
callbacks.model_checkpoint.auto_insert_metric_name=false \
'callbacks.model_checkpoint.filename="epoch_{epoch}-step_{step}"'
~~~

上述保存参数保留第 100、200 轮的完整 checkpoint（epoch 字段分别为 99、199），
`save_top_k=-1` 避免第 200 轮删除第 100 轮；`last.ckpt` 仅随这两个保存事件更新。
优化器、调度器和 epoch 从头建立，不继承 Stage-1 状态。
训练后的 checkpoint 已包含八桥配置和学到的投影；推理、严格重载及同结构断点续训
不会再次 zero-init。续训时关闭两个迁移开关，并使用 `resume_ckpt_path`。

~~~bash
python -m pytest tests/test_g1_bridge_expansion.py tests/test_g1_checkpoint_expansion.py -q
~~~

## 9. 当前架构与参数

后续修改以本节方案为基础。六个子模块可独立修改，接口由 G1Module 统一检查。

- 条件编码：c 为 256→128；同构件有效原子对 p 为 128→64；1 层、4 头静态 Transformer，
  输出 128 维，不注入当前噪声和时间调制。
- 四桥固定共享一套 ConditionEncoder，一次 Flow forward 只编码一次。
- Router、组交互和原子读回在各桥独立学习；每桥使用当前 H/t 重新计算。
- K_b=min(max_groups,N_b)，默认上限 32；分组布局在同次前向内复用，不共享学习得到的 P/Pbar。
- 当前周期几何默认四桥复用；组内几何描述只在同一桥内的组交互和读回之间复用。
- 默认主干、桥、条件编码器和几何模块均关闭激活重算。重算仍是可调的内存功能。

| 范围（small：token_s=256、token_z=128） | 参数量 |
| --- | ---: |
| 共享 ConditionEncoder（合计一套） | 374,016 |
| 每桥独立的 Router | 60,128 |
| 每桥独立的组交互 | 231,876 |
| 每桥独立的原子读回 | 151,716 |
| 周期几何与 U/Z 汇总 | 0 |
| G1 合计（共享条件 + 四套更新） | 2,148,896 |
| 原始 AtomMOF small | 38,376,320 |
| **AtomMOF small + G1** | **40,525,216** |
| AtomMOF small + G1，四桥 K64 | 40,529,312 |
| AtomMOF small + G1，八桥 K64（每三层插入） | 42,308,288 |

当前训练接口不再提供 share_condition 或 condition_encoder.pair_mode 两个参数；
四桥共享与紧凑 pair 已是唯一实现。不要在新命令中继续传入这两个配置键。
外部 dense c/p 的 encode 入口、缺少 group_layout 的旧 G1Context 接口也已删除。
GroupInternalGeometry 及其提取函数统一位于 geometry/groups.py。

## 10. 阶段一训练配置

configs/experiment/g1_small_stage1.yaml 组合当前方案：

- 8 GPU、FP32、单卡动态 batch 上限 32、梯度累积 2。
- small token 宽度 256/128、最多 200 原子、完整筛选后训练集、验证 1000 样本、500 epochs。
- 四桥位置 6/12/18/24；共享条件、默认共享几何；所有激活重算关闭。
- AF3 学习率调度、max_lr=1e-4、warmup=10000 次优化器更新；默认从头训练。

在已激活环境的服务器上，新建运行目录：

~~~bash
cd /data/ai4science/AtomMOF-G1 || exit 1
export PROJECT_ROOT=/data/ai4science/AtomMOF-G1
export RUN_DIR="$PROJECT_ROOT/logs/train_bwdb_g1_small_stage1_8gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u src/train.py \
  experiment=g1_small_stage1 \
  data.lmdb_dir=/data/ai4science/AtomMOF/AtomMOF-main/data/bwdb \
  hydra.run.dir="$RUN_DIR" \
  logger=wandb \
  +logger.wandb.entity=moonshuo-42-meta \
  logger.wandb.project=AtomMOF-9 \
  logger.wandb.offline=false \
  2>&1 | tee "$RUN_DIR/train.log"
~~~

本次清理后生成的 checkpoint 可按相同结构续训。清理前 checkpoint 中保存的
share_condition/pair_mode 等已删除字段不再接受，也不自动迁移旧配置。
修改磁盘代码不会改变已运行 Python 进程；本次本地清理不改变服务器上正在执行的实验。

## 11. 当前计算实现与性能验证

### 条件原子对

只对同构件有效原子对计算 p(M,128)、原 pair MLP、128→64 投影和每头 bias，
M 为各样本各构件原子数平方之和。只有 bias 散射回 (B,heads,N,N)；
不构造完整的 128/64 通道原子对张量。QK/softmax 仍为稠密 N×N。
这是当前唯一训练路径；dense 数学参考仅在 tests/reference_g1_condition.py，
由数值测试和组件基准使用，训练与推理不导入该文件。

### 几何与分组复用

geometry/planning.py 批量规划搜索边界，按本批实际最大候选数分配镜像槽。
默认 pair_chunk_size=32768、image_chunk_size=524288，按镜像数同时限制分块规模；
保留全部合法候选、125 候选上限和超限样本整条禁用几何的规则。
这些是当前几何定义与资源控制，不是旧模型的兼容逻辑。

G1Bridges.prepare 按 max_groups 复用 GroupLayout；它是 G1Context 的必需字段。
缓存只存在于本次前向，不跨 batch 或优化器更新。
geometry/groups.py 每桥提取一次 Z_kk、chi(Z_kk) 和 Q_kk，供组交互和读回共享；
不同桥的 Z 不同，不能共享该描述。完整对称矩阵收缩和可微路径保留。

### 组件基准

与正式训练分开运行：

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_g1_optimizations.py \
  --device cuda --batch-size 32 --atoms 200 --block-size 20 \
  --warmup 5 --iterations 20 --backward \
  --output logs/g1_component_benchmark.json
~~~

脚本先核对输出，再预热并同步计时。它调用测试目录的数学参考进行对照，
不会把参考路径注册到训练配置。组件时间不能直接换算成八卡 epoch 时间；
完整训练需统一数据、batch、精度、预热和验证口径后测量。

## 12. 历史记录

历史实现、完整旧命令和当时的验证输出通过 Git 查询，不在当前文档中维护多套操作方案。

| 提交 | 当时的变更 |
| --- | --- |
| 9a971b0 | 可选 G1 四桥接入训练与推理 |
| 8b8e551 | 条件编码先降维，再经过一层 Transformer |
| 40a5290 | 共享条件编码、批量周期几何、默认关闭重算 |
| e41eb75 | 紧凑原子对、分组布局与组内描述复用、几何分块优化 |

2026-10-16 清理：只维护当前共享/紧凑方案，删除旧 G1 兼容入口，
合并当前说明，并将 dense 数学参考移至测试目录。既有 Git 历史保留。
