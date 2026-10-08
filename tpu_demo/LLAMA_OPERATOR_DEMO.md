# GPTQ / W4A16（算子三）RV 迁移问题

> 芯片：**SG2260E**（BM1690 同族）
> 分支：`feature/sg2260e-rv-support`
> SDK：PPL v1.7.122

---

## 一、问题

**GPTQ / W4A16 算子在 SG2260E 的 RV 通路上跑不起来。**

根因：这颗芯片上负责"4-bit 权重分组反量化"的原语（`tiu::dq2` → `rvt_dq2` / `tpu_bdc_f16_group_dequant`）**在两个后端上都没有可用实现**。

不是我们的接入问题 —— 是这条硬件通路在 RV 侧不存在，TPU-Kernel 侧只有桩。

---

## 二、阻塞证据链

### ① 厂商自己的参考实现就编不过

用 PPL SDK 重编 `examples/cxx/matmul/w4a16_matmul_dq2.pl`，两个 dq2 调用点（`right_slice` 真/假两个分支，覆盖**全部**路径）都生成无条件断言：

```c
TPUKERNEL_ASSERT( 0 && "Chip have no instruction like DQ2Op!!!\n");
```

`--chip=sg2260e`（tpukernel）与 `--chip=sg2260erv --rv` **两者都断言**。

### ② 固件里有符号，但手工发射会把卡挂死

`rvt_dq2` 声明在 `deps/chip/tpub_7_1_e/TPU1686/kernel/include/rvt_api.h`，且在 `libfirmware_core.a` 里有定义（`nm` 显示 `T`）。

**但手工发射它会卡死在 `rt_task_done_response_sync`**，必须 `sudo rmmod sgcard && sudo modprobe sgcard` 才能恢复。符号存在，后面是空的。

### ③ 厂商把操作数准备完了，却拒绝发这条指令

断言前一行：

```c
ppl_tensor_t right_local_u4_v175 = {..., .dtype = DT_UINT4, .align_mode = 1, ...};
rvt_tr(8, PRECISION(DT_UINT4), SIGN(DT_UINT4), right_local_u4_v175.addr,
       HW_ALIGN_LAYOUT, (array4_t){...}, (int *)NULL);
TPUKERNEL_ASSERT( 0 && "Chip have no instruction like DQ2Op!!!\n");
```

**寄存器 8 正是 `rvt_dq2` 的 `a` 操作数** —— 描述符构造好、装进去了，然后拒绝发指令。这不是运行时检查，是厂商后端映射表的缺项。

### ④ TPU-Kernel 侧的分组反量化在硬件上是桩

`tpu_bdc_f16_group_dequant`：

| 位置 | 符号类型 |
|---|---|
| `libfirmware_core.a:nodechip_llama_mlp_multi_core.c.o` | `U`（固件自己的 Llama MLP 引用它） |
| `libfirmware_core.a:tpu_kernel_common.c.o` | **`W` 弱符号，110 字节** |
| `libtpuv7_emulator.so` | `T` 强定义 |

**只有模拟器里有真实现，硬件侧是 110 字节的弱桩**（真正的分组反量化实现不可能只有这个体量）。

### ⑤ 厂商自己的 RV 编译器从不发射任何 `rvt_dq*`

统计厂商**自己**为 sg2260erv 生成的 C 代码（`paged_attention_multicore.c`）中的指令出现次数：

| 指令 | 次数 |
|---|---|
| `rvt_and` / `rvt_cvt_i2i` / `rvt_cvt_i2f` | 1 / 1 / 1 |
| `rvt_mul` / `rvt_sub` | 2 / 1 |
| **`rvt_dq*`** | **0** |

同样，`w4a16_matmul_dq2.pl` 的 RV 产物里 `rvt_dq*` 也是 **0** 次 —— 所有 dq2 调用点都换成了断言。

**结论：RV 层这条路不通，不是"还没映射"。**

---

## 三、附带发现：手册 4.6 的 cast 转型表与真机不符

排查替代方案时发现，手册 `ppl::tiu::cast` 的转型表在真机上不成立。手册把 INT8/UINT8 作**源**的两行标为空，且没有 INT4/UINT4 的行或列。

**真机实测**（SG2260E，PPL harness，PCIe），设备输出 vs 参考模型：

| 用例 | 手册预期 | 真机结果 |
|---|---|---|
| `fp32 → fp16`（对照） | 支持 | ✅ 0/4096 不匹配 |
| `int32 → fp16` / `int16 → fp16` | 支持 | ✅ 0/4096 |
| **`int8 → fp16`** | **不支持** | **✅ 0/4096（手册错误）** |
| `fp16 → int8` | 支持 | ❌ 挂卡 |
| `int8 → int8`（**纯拷贝，无 cast**） | — | ❌ 挂卡 |
| `int4 / uint4 → fp16` | 不支持 | ⏸ 未测到（见 §4） |

`int8 → fp16` 是真正的整数→浮点转换，不是字节重解释：输入 int8 的 `27` 若按字节重解释为 fp16 会得到 ~2.6e-5 的次正规数，实测得到 **27.0**。

另：`tiu::cast` 在头文件里是**完全无约束的模板**，`ppl-compile` 对 dtype 组合**不做任何合法性校验** —— 含 4-bit 双向在内的 9 个用例全部 `rc=0` 通过，且 16/8/4 位整数源发射**同一形式**的指令。

**所以编译通过不构成证据，唯一判据是真机行为。**

---

## 四、附带发现：8 位整数作为全局缓冲区会挂卡

**纯拷贝 `int8 → int8`（一行 cast 都没有）同样挂卡** —— 说明挂卡出在 **8 位整数作为全局缓冲区**的 I/O 路径上，与 cast 指令的 dtype 支持无关。

拆开看是两个独立问题：

- **8 位整数作全局输出** → `fp16→int8`、`int8→int8` 均挂
- **uint8 作全局输入** → `uint8→fp16` 挂（但 `int8→fp16` 正常）

挂卡特征：进程进入 `D`（不可中断睡眠），wchan 为 `send_quit_request`，`timeout` 与 `SIGTERM` 均无法终止，**需 `sudo rmmod sgcard && sudo modprobe sgcard`**。

这直接限制了低精度实验的能力：**`int4 / uint4 → fp16` 至今未能测到**，属"方法尚未建立"，而非"已证明不支持"。

---

## 五、需要确认 / 协助的事项

| # | 事项 |
|---|---|
| 1 | **SG2260E 的 RV 通路上能否补齐 4-bit 分组反量化。** 若不能，请明确 4-bit 量化权重在这颗芯片上的推荐做法 |
| 2 | **4-bit 拆包能否用现有基础指令组合实现。** 若能，我们可以自己拼出反量化，不必依赖 `rvt_dq2` |
| 3 | **修正手册 4.6 数据类型转换表的错误**（`int8 → fp16` 实际可用） |
| 4 | **排查 8 位整数作全局缓冲区导致挂卡的问题** |

---

## 六、如何验证已经修好

**验收用例已入库**：[`gptq_w4a16/gptq_w4a16.py`](gptq_w4a16/gptq_w4a16.py)

用例自带一个**主机侧 oracle**（torch 实现的语义参考）。跑完把设备输出与 oracle 逐元素比对，返回机器可读的 JSON：

| `status` | 含义 |
|---|---|
| **`passed`** | 编译通过 → 上板跑通 → 与 oracle 在容差内一致。**这就是"修好了"。** |
| **`unsupported`** | 工具链拒绝降级该原语，`evidence` 带编译器原文。**这是"还没好"，且是可复现的、非断言的记录。** |
| 抛异常 | 跑通了但数值不符，或环境问题 |

### 跑法

```bash
# 矩阵 runner（带看门狗、首次失败即停）
python testing/python/jit/tpu_demo_ops_matrix.py \
    --runtime-mode pcie --chip sg2260e --programming-model rv \
    --case gptq-w4a16.float16 --device-id 0 --allow-pcie --allow-pcie-profile \
    --output-dir /tmp/matrix
```

修好之后 `status` 应变为 `passed`，形如：

```json
{
  "status": "passed",
  "operation": "gptq-w4a16", "dtype": "float16",
  "chip": "sg2260e", "programming_model": "rv",
  "metrics": {"passed": true, "atol": 0.01, "rtol": 0.01,
              "max_abs_error": 0.0009765625,
              "mismatched_elements": 0, "element_count": 2048}
}
```

**整个过程不需要我方参与，也不需要改动任何代码。**

### 当前状态（改之前）

| 后端 | 表现 |
|---|---|
| **RV Tensor** | 编译期拒绝，用例返回 `status: unsupported`，`evidence.message` 为编译器原文 |
| **TPU-Kernel** | 能编译（发射 `tpu_bdc_f16_group_dequant`），但该符号在固件里是桩。未做上板验证 |

> ⚠️ **已知障碍**：目标缺 lowering 时，用例在 **cmodel 下干净返回 `unsupported`**，但在 **PCIe + profile 环境下会 SIGSEGV** —— TVM 的致命错误路径抓 backtrace 时自己崩了（gdb 栈落在 `_Unwind_Backtrace` → `backtrace_syminfo` → `__memmove_avx512_unaligned_erms`）。`TVM_BACKTRACE=0` 无效。**目前 `unsupported` 这个结论只能在 CModel 下拿到。**
>
> runner 的 `--runtime-mode pcie` 还要求先有 `--bm-cmodel-summary` / `--sg-cmodel-summary` 的 CModel 阶段产物，而 **CModel 需要 root**。

---

## 七、相关文件在仓库中的位置

### 用例与判定

| 文件 | 仓库位置 | 说明 |
|---|---|---|
| GPTQ 验收用例 | [`tpu_demo/gptq_w4a16/gptq_w4a16.py`](gptq_w4a16/gptq_w4a16.py) | 算子本体 + 主机 oracle + 三态判定 |
| 用例注册表 | [`tpu_demo/cases.py`](cases.py) | `--case` 的取值来源 |
| 公共判定与容差 | [`tpu_demo/common.py`](common.py) | `DemoUnsupported` / `unsupported_payload` / 容差家族 |
| 矩阵 runner | [`testing/python/jit/tpu_demo_ops_matrix.py`](../testing/python/jit/tpu_demo_ops_matrix.py) | 带看门狗的上板入口 |
| 算子-指令映射契约 | [`tpu_demo/OP_MAPPING.md`](OP_MAPPING.md) | 各算子到指令的映射 |

### 算子三相关的实现

| 文件 | 仓库位置 | 说明 |
|---|---|---|
| `ppl_dq2` 前端 contract | [`tilelang/language/customize.py`](../tilelang/language/customize.py) | 参数校验 + 语义说明 |
| `tl.tpu.dq2` 的 codegen | [`src/target/codegen_tpukernel.cc`](../src/target/codegen_tpukernel.cc) | **含 RV 路径的显式拒绝**（`ICHECK_EQ(target_programming_model_, "tpukernel")`） |

### 不在仓库中的证据（本地）

以下证据文件目前**不在 git 仓库里**，需要的话可以一并入库：

| 用途 | 本地路径 |
|---|---|
| cast dtype 阶梯探针（编译期 9 用例） | `dq2-verify/cast_probe/cast_ladder.py` |
| cast 真机验证 | `dq2-verify/cast_hw/run_case.sh` |
| 厂商 `.pl` 的 RV 编译产物（含断言原文） | `dq2-verify/rv_compile/`、`dq2-verify/tpk_compile/` |
| W4A16 TileLang 侧早期尝试 | `dq2-verify/w4a16_matmul_kernel.py` |

---

## 八、其余三个算子的验收

其余三个算子（RMSNorm / Llama MLP / Paged Attention）**当前均已在 SG2260E 上通过**，判据与命令和算子三完全相同 —— **只需换 `--case`**。

| 算子 | 用例 | 实现（仓库位置） |
|---|---|---|
| 一、RMSNorm | `rmsnorm.float16`、`rmsnorm.bfloat16` | [`rmsnorm/rmsnorm.py`](rmsnorm/rmsnorm.py) |
| 二、Llama MLP | `llama-mlp.float16`、`llama-mlp.bfloat16` | [`llama_mlp/llama_mlp.py`](llama_mlp/llama_mlp.py) |
| 四、Paged Attention | `paged-attention.float16`、`paged-attention.bfloat16` | [`paged_attention/paged_attention.py`](paged_attention/paged_attention.py) |

```bash
python testing/python/jit/tpu_demo_ops_matrix.py \
    --runtime-mode pcie --chip sg2260e --programming-model rv \
    --case llama-mlp.float16 --device-id 0 --allow-pcie --allow-pcie-profile \
    --output-dir /tmp/matrix
```

**已实测结果**（SG2260E，PCIe）：

| 用例 | status | 不匹配 | 最大误差 | 容差 |
|---|---|---|---|---|
| `llama-mlp.float16` | ✅ passed | 0/2048 | 9.8e-4 | 1e-2 |
| `llama-mlp.bfloat16` | ✅ passed | 0/2048 | 3.9e-3 | 3e-2 |
| `paged-attention.float16` | ✅ passed | 0/256 | 1.2e-4 | 2e-2 |
| `paged-attention.bfloat16` | ✅ passed | 0/256 | 9.8e-4 | 2e-2 |

> **两个容易混淆的对应关系**：
> `swiglu/` **不是**算子二 —— 它只是 MLP 里的 SwiGLU 激活，没有 gate/up/down 三个投影。
> `flashattn/` **不是**算子四 —— 那个是非分页的，没有 KV cache 和 block table。
