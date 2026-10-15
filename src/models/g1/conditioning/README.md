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

参数默认保持 small 编码器的单特征 256 维、pair 128 维、3 层、4 头，
末端投影到 G1 条件接口的 128 维。`output_dim=None` 可取消末端投影。
独立 Hydra 配置位于 `configs/g1/condition_encoder/static.yaml`。

## 模块划分

- `features.py / AtomConditionFeatures`：复现原 `AtomAttentionEncoder` 的 c、p
  公式、参数命名和初始化，包含原版 single-to-pair 与 pair MLP。
- `attention.py / BlockPairAttention`：复用原 attention 的 Q/K/V、pair bias、
  特征 sigmoid gate 和计算后端，仅增加严格同构件 query/key 掩码。
- `encoder.py / StaticConditionBlock`：移除时间嵌入、adaLN shift/scale 与时间 gate，
  保留原 LayerNorm 设置及 attention/MLP 的普通残差。
- `encoder.py / ConditionEncoder`：组织特征、固定 p 的 Transformer 层和输出投影。

同构件的非成键原子也能直接交互。有效原子永远不能读取 padding；padding
query 暂时只允许读取自己以避免全屏蔽 softmax，随后输出清零。全 padding 样本
也返回有限的零张量。输入沿用原 collator 的有限、零填充约定。

## 与原版一致的范围

仅复用计算方式，参数默认重新初始化，不自动加载官方权重。
原始生成主干和现有 checkpoint 键保持不变。
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

这个入口不调用 `encoder.features`；若始终使用外部 c/p，应相应管理未使用的
特征模块参数。重复候选可以在编码后 `repeat_interleave`，无需重复构造条件。
同一固定条件的采样过程可缓存输出；训练时保持梯度，不能跨优化器更新复用旧缓存。

## 计算与验证

当前仍使用 dense pair 张量和 attention；加 mask 不会自动获得按构件拆分的
计算节省。后续可以独立替换 attention 后端或特征构造。

```bash
python -m pytest tests/test_condition_encoder.py -q
```

测试覆盖原始 c/p 数值对齐、取消时间调制后的残差、跨构件隔离与梯度、
非成键远端交互、padding/单原子、原子置换与构件重编号、不读取目标或噪声、
manual/PyTorch 后端一致性和 activation checkpointing。

六模块统一封装与调用方式见 [G1Module](../README.md)。


当前主干接入、开关、checkpoint 兼容性与 263 项全仓库测试记录见 [G1 总文档第 8 节](../README.md#8-atommof-主干接入与开关)。
