# G1 静态条件编码器

按讨论方案实现的独立模块，现已通过 G1Module/G1Bridges 接入 Flow 主干，默认关闭。

```python
from src.models.g1.conditioning import ConditionEncoder

encoder = ConditionEncoder()  # 构造一次，并注册为上层模型的子模块
c_cond = encoder(feats)       # (B, N, 128)，padding 行为 0
```

`feats` 使用原始 BWDB collate 的字段和单位：

| 字段 | 形状 | 用途 |
| --- | --- | --- |
| `ref_pos` | B,N,3 | 构件参考坐标，沿用原版 Å 单位 |
| `ref_element`, `ref_charge`, `block_type` | B,N | 原子条件 |
| `atom_pad_mask` | B,N | 1/True 为有效原子 |
| `block_index` | B,N | 仅用于同构件关系，不编码编号大小 |
| `bond_types` | B,N,N | 原版稠密键类型矩阵 |

默认结构（2026-10-16 更新）：

~~~text
原始条件特征构造
c(B,N,256)   ── Linear(256,128,bias=False) ── c'(B,N,128) ─┐
p(B,N,N,128) ── Linear(128,64,bias=False)  ── p'(B,N,N,64) ─┤
                                                       ↓
                         1 层静态 Transformer（4 头，FFN 比例 4）
                                                       ↓
                                             c_cond(B,N,128)
~~~

先降维，再进入 Transformer。原始 c/p 构造公式、同构件 attention 掩码和无时间调制的
残差保持一致，默认不再需要末端降维。后续 Router 等模块的 128 维条件接口不变。

独立 Hydra 配置位于 `configs/g1/condition_encoder/static.yaml`：

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `atom_s`, `atom_z` | 256, 128 | 原始 c/p 构造宽度 |
| `hidden_dim`, `pair_dim` | 128, 64 | Transformer 的 single/pair 宽度 |
| `depth`, `heads` | 1, 4 | Transformer 层数与头数 |
| `output_dim` | 128 | 输出接口宽度 |

这些宽度独立配置；只有 `hidden_dim` 必须能被 `heads` 整除。
`output_dim=None` 表示输出 `hidden_dim` 维；输出宽度相同时使用 Identity。
入口宽度和内部宽度相同也使用 Identity，便于后续切换压缩方案。

## 模块划分

- `features.py / AtomConditionFeatures`：复现原 `AtomAttentionEncoder` 的 c、p
  公式、参数命名和初始化，包含原版 single-to-pair 与 pair MLP。
- `attention.py / BlockPairAttention`：复用原 attention 的 Q/K/V、pair bias、
  特征 sigmoid gate 和计算后端，仅增加严格同构件 query/key 掩码。
- `encoder.py / StaticConditionBlock`：移除时间嵌入、adaLN shift/scale 与时间 gate，
  保留原 LayerNorm 设置及 attention/MLP 的普通残差。
- `encoder.py / ConditionEncoder`：组织特征、独立的 `single_projection` /
  `pair_projection`、固定压缩后 p 的 Transformer 层，以及可选输出投影。

同构件的非成键原子也能直接交互。有效原子永远不能读取 padding；padding
query 暂时只允许读取自己以避免全屏蔽 softmax，随后输出清零。全 padding 样本
也返回有限的零张量。输入沿用原 collator 的有限、零填充约定。

## 与原版一致的范围

仅复用计算方式，参数默认重新初始化，不自动加载官方权重。
原始 AtomMOF 主干及 baseline checkpoint 的键和形状保持不变。
旧 G1 三层条件编码器与本版的参数键、形状不同，不能直接用其 checkpoint 恢复本版训练。
新 G1 checkpoint 会保存 `hidden_dim` / `pair_dim` / `depth`，可按本版结构恢复；
从 baseline 权重初始化 G1 的入口保持不变。
`ref_chirality` 仍未显式使用；不新增 RBF、最短路径或构件编号 embedding。
保留原始 `ref_pos` 和参考位移，因此不宣称参考构象旋转不变。
原版键 embedding 和 single-to-pair 项不全部乘同构件 mask；禁止跨构件读取
由 attention 分数上的硬掩码实现。

如后续需要复用已有编码器的条件参数，可显式复制同名参数：

```python
source = existing_atom_encoder.state_dict()  # 已有 AtomAttentionEncoder 实例
encoder.features.load_state_dict(
    {name: source[name] for name in encoder.features.state_dict()}, strict=True
)
```

也可直接传入已有 c、p，使用相同设备、维度及原始特征约定：

```python
c_cond = encoder.encode(c, p, atom_pad_mask, block_index)
```

这个入口接收原始宽度的 c/p，并在内部执行两项入口投影；不调用 `encoder.features`。
若始终使用外部 c/p，应相应管理未使用的
特征模块参数。重复候选可以在编码后 `repeat_interleave`，无需重复构造条件。
同一固定条件的采样过程可缓存输出；训练时保持梯度，不能跨优化器更新复用旧缓存。

## 计算与验证

默认 pair_mode=local：先收集同构件有效原子对，再按原公式构造紧凑的
p(M,128)，经过原 pair MLP 和 128→64 投影，M=各样本各构件原子数平方之和。
Transformer 内 pair LayerNorm 和 bias 投影也只在这些原子对上运行，
最后散射为 (B,heads,N,N) 注意力偏置。避免分配 dense p 的 128/64 通道张量。

attention 的 QK 和 softmax 仍是 dense N×N，不能把整层复杂度都算成 M。
布局压紧每次条件编码包含一次 nonzero；训练不跨 optimizer step 缓存。
pair_mode=dense 保留完整参考路径；features(feats) 和 encode(c,p,mask,block_index)
的公开 dense 接口仍可使用。无有效原子对时所有参数仍连接到零梯度图，支持 DDP。
数学公式和权重键/形状保持一致；GEMM 与梯度求和顺序变化允许浮点舍入误差。
GPU 加速比例与构件大小分布有关，需实测。

单个默认 ConditionEncoder 参数量：

| 部分 | 参数量 |
| --- | ---: |
| 原始 c/p 特征构造 | 118,656 |
| c 入口投影 256→128 | 32,768 |
| p 入口投影 128→64 | 8,192 |
| 1 层 Transformer（attention + FFN） | 214,400 |
| 输出 Identity | 0 |
| **合计** | **374,016** |

之前三层编码器每个为 2,715,776 参数，精简后每个为 374,016。
当前多桥配置 share_condition=true：四桥共享同一个编码器实例及一次前向输出，总共只训练 374,016 个条件参数。
独立 G1Module() 仍各自创建自己的编码器；多桥共享由 G1Bridges 管理。

```bash
python -m pytest tests/test_condition_encoder.py -q
```

测试覆盖原始 c/p 数值对齐、取消时间调制后的残差、跨构件隔离与梯度、
非成键远端交互、padding/单原子、原子置换与构件重编号、不读取目标或噪声、
manual/PyTorch 后端一致性、入口降维的形状和梯度、可配置宽度、参数预算和 activation checkpointing。

六模块统一封装与调用方式见 [G1Module](../README.md)。


主干接入与开关见 [G1 总文档第 8 节](../README.md#8-atommof-主干接入与开关)，
精简记录见第 9 节；最新共享、参数量和阶段一命令见 [G1 总文档第 10 节](../README.md#10-共享条件批量几何与关闭重算2026-10-16)。
