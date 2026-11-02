# GPTQ / W4A16：BM1690E RV 软件反量化

当前上游统一使用 `BM1690E` / `bm1690e` 命名；下文历史验证记录中的
SG2260E 对应同一 `tpub_7_1_e` SDK 架构，历史手册名称保持原样。

SG2260E 指令集不支持 DQ2。RV 后端通过普通整数及浮点指令实现
`ppl_dq2`，不发射 `rvt_dq2`，也不调用 TPU-Kernel 的反量化符号。
这解决的是编译器和 CModel 上的软件实现；**尚未完成 PCIe 验收**。

## 数据与数值契约

`T.ppl_dq2(out, src_packed, offset_scale, gsize)` 接受互不别名的 rank-2
local tensor：

- `out`：FP16/BF16，形状 `(N,K)`。
- `src_packed`：UINT8、UINT16 或 UINT32，每元素含 2、4 或 8 个无符号
  4-bit code，最低有效 nibble 在前。宽度为 `K / codes_per_word`。
- `offset_scale`：UINT32，形状 `(N,K/gsize)`；低 16 位是 FP16 offset，
  高 16 位是 FP16 scale 的**位模式**。BF16 输出也使用 FP16 参数。
- `gsize`：正整数，整除 K。RV 支持组边界落在 packed word 内部。

RV 在 FP32 中分别执行 `(q - offset)` 和乘 scale，最后以 nearest-even
转换到输出类型一次；不做饱和，保留 NaN/Inf 和溢出到 Inf 的行为。
GEMM 使用已经转换到输出类型的权重。主机 oracle 同样显式执行这个
中间舍入，不能拿未经 FP16/BF16 舍入的权重冒充设备 GEMM 的输入。

## 指令序列与布局

依据《SG2260E RV 架构手册》§3.6.9.1（第 42 页）、§3.6.10.1
（第 44 页）、§3.6.14（第 51–52 页）及 §3.1.4（第 6 页）：

1. 把当前 packed word 复制或零扩展到 UINT32 scratch。
2. `rvt_lshr` 使用 S8 **负移位量**，右移 `4 * nibble_index`。
   设置 `rvt_cfg_round_mode(2)`（向零舍入），防止默认 nearest-even
   在右移时改变高 nibble；随后 `rvt_and(..., 0xF)`。
3. 恢复 `round_mode(0)`，用 `rvt_cvt_i2f` 转为 FP32。
4. 将参数表的两个半字以 FP16 描述符解释，并转为 FP32。
5. `rvt_fsub`、`rvt_fmul`，最后 `rvt_cvt_f2f` 写入对应输出列。

各列并行处理 N 个通道，顺序写入解包后的 K 列，避免额外交错搬运。
使用固定的 CR1/CR2、TR8–TR13；遵守现有同步边界，不启用异步流水线。
工作区是显式的 `(N,16)` FP32 tile，仅前四列存放中间结果。完整 tile
纳入读写效果、活跃区间及 LMEM 分配；源权重和参数表只读。
这是正确性优先的逐列实现，**没有性能验收或吞吐量承诺**。

RV demo 默认使用 UINT32 打包，以避免已报告的 UINT8 全局 I/O 路径。
每个 UINT32 的八个 code 按低位在前排列；外部 UINT8 权重可在主机
按字节重新打包为 UINT32，无需在主机反量化。旧 UINT8 local 契约仍保留。
改变打包类型不等于已证明硬件安全。

## 验证入口

从仓库根目录，使用匹配此源码的独立 native build 和 PPL v1.7.122
完整开发包。将仓库加入 `PYTHONPATH`，并确保 `python` 指向该环境。
最新上游要求通过隔离 launcher 运行，不能直接使用共享 `/tmp`；需要
Linux user namespace 和 bubblewrap。`tpu_demo.run` **只运行 CModel**：

```bash
python -m pytest -q testing/python/jit/test_tpu_rv_dq2.py

# 从仓库根目录执行；每轮使用新的绝对结果目录。
repo_root="$PWD"
run_dir="$PWD/research/artifacts/dq2-$(date +%Y%m%d-%H%M%S)"
for case in $(seq 0 47); do
  python -m tilelang.jit.tpu_cmodel --output-dir "$run_dir/dq-$case" -- \
    python "$repo_root/testing/python/jit/test_tpu_rv_dq2.py" --case "$case" || exit 1
done
for case in $(seq 0 11); do
  python -m tilelang.jit.tpu_cmodel --output-dir "$run_dir/gptq-$case" -- \
    python "$repo_root/testing/python/jit/test_tpu_rv_dq2.py" --gptq-case "$case" || exit 1
done
for dtype in float16 bfloat16; do
  python -m tilelang.jit.tpu_cmodel --output-dir "$run_dir/demo-$dtype" -- \
    python -m tpu_demo.run --case "gptq-w4a16.$dtype" \
      --chip bm1690e --programming-model rv || exit 1
done
```

48 项定向测试覆盖三种 packing × 两种输出 dtype × 八组配置：全部
256 种字节值、1/63/64/65/129 行、8–4096 列、组大小 1/3/4/32/64/128/256、
负数及小数 offset/scale、NaN/Inf/溢出；重复调用检查输入未被破坏。
要求反量化输出与独立主机参考逐元素相等（NaN 位置相等），不放宽容差。
另有 native scratch 拒绝测试、注册表对齐和 LMEM 活跃存储不重叠测试。

### 本轮实际结果（2026-10-19 执行，2026-10-21 整理）

基线为 `e31b98ec4cdb74d33d5663d0d9512b71fcfb9b98`，固定 TVM
`a8a54d2b1f43c23a47f2fc08779654918eae6464`，独立构建 native 库，使用
PPL `v1.7.122-g05ebfb36-20260528`，未复用旧工作树 native 产物。

- SG2260E RV 反量化：**48/48**；共 183408 个输出元素，有限值误差为零，
  NaN 位置一致。
- SG2260E RV 反量化 + GEMM：**12/12**；三种 packing、FP16/BF16、
  单 tile 和多 tile，沿用 matmul 的原有容差。
- 默认 GPTQ RV demo：FP16/BF16 **2/2**，均返回 `passed`，无不匹配元素。
- 前端、描述符、残余 IR、注册表、LMEM、demo/core/FP8 矩阵与 RVT
  静态检查：**686 passed，6 deselected**；六项 PCIe 编译链接检查依赖
  本机缺失的 `libtpuv7_rt.so`，未用 emulator 替代。随后新增默认 packing
  一致性测试，单独复跑 DQ2 文件 **14/14**（其余 13 项包含在前述 686 项内）。
- 额外尝试 BM1690 TPU-Kernel FP16 GPTQ CModel 时，SDK 的
  `tpub_7_1/lib/libtpuv7_emulator.so` 在 `tpu_bdc_f16_group_dequant` 内
  报 `is not supported on SG2260` 并断言。后续 TPU-Kernel 数值测试停止；
  不能把模拟器含有该符号解释为它可用。本分支未改用该路径实现 RV。

以上是定向 CModel 验证，**不是 BM/SG 全部 79 项 demo 的完整 promotion
矩阵，也不是 PCIe 或性能验收**。未连接板卡、未执行 reset 或驱动操作。

### 2026-10-27 rebase 验证边界

本分支已适配上游 `cd8356559fd636aa51d91c7c41ebafe76c557626`，保留
BM1690E 命名、新 scatter 行数参数、并行命令注册及新版地址分配。
DQ2 的 scratch 读写合入新分配器的实际分派分支，避免重复分支漏记。

native 重建成功；DQ2、前端契约、描述符、语义注册、残余 IR 及地址分配
测试 **257 passed**。尝试经上游私有 IPC launcher 复测 CModel 时，
bubblewrap 在挂载 `/proc` 阶段返回 `Operation not permitted`，未进入
数值 launch；没有绕过隔离要求。上面的 62 项数值通过记录只属于旧基线，
不能作为本次 rebase 的数值验收。需在允许该 namespace/mount 操作的
环境重新运行上述 CModel 命令。本次没有进行 PCIe 测试。

## 硬件限制及后续验收

此前问题记录见 [issue #48](https://github.com/TongmingLAIC/tilelang-tpu/issues/48)：

- 厂商 PPL 的 W4A16 DQ2 示例生成 `Chip have no instruction like DQ2Op`
  断言；头文件中的 `rvt_dq2` 声明不能作为支持依据。
- 手工发射 `rvt_dq2` 曾挂卡，禁止再次试探。
- UINT8 全局输入及 INT8 全局输出曾有挂卡报告。软件拆包没有修复 DMA
  或驱动问题；UINT8 CModel 通过不代表该 I/O 在真机安全。
- TPU-Kernel 仍仅接受 UINT8 packing 和组大小 32/64/128/256，沿用
  `tpu_bdc_f16_group_dequant`。此前记录其硬件符号是弱桩；不能通过切换
  后端把它当成硬件可用的替代方案。

任何 PCIe 验证必须先完整通过匹配源码/native/SDK 身份的 BM1690 和
SG2260E CModel 矩阵，再使用受监督 runner、共享板卡锁、session 和
quarantine 协议。新整数指令组合的硬件行为尚未验收，执行前应由维护者
协调可恢复板卡的人员；不能把杀掉 host 进程当作板卡恢复。

## 其他模型算子

Llama MLP、paged attention 的 FP16/BF16 PCIe 通过记录来自上游此前
验证，本次 DQ2 定向测试不重新证明这些结果，也不构成完整模型验收。
对应入口分别在 `tpu_demo/llama_mlp/`、`tpu_demo/paged_attention/`；
通用矩阵和 profiling 流程以 `tpu_demo/README.md` 为准。
