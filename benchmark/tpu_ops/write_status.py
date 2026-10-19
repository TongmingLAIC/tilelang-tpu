"""Write concise results plus complete per-operator times and ablation tables."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from .workloads import cases, case_sort_key, OPERATOR_ORDER

LABELS = {
    'ppl': 'PPL',
    'tilelang': '原始 TL',
    'tilelang-matched': 'PPL 算法 TL',
    'tilelang-optimized': '新优化表达 TL'
}
STATUS_LABELS = {
    'passed': '通过',
    'not_run': '未执行',
    'numerical_mismatch': '数值不符',
    'timeout': '超时',
    'error': '运行错误',
    'launch_failed': '进程退出，未获数值结论',
    'measured': '已有有效计时',
    'unsupported': '当前通路不支持',
    'failed': 'CModel 未通过',
    'pending': '待 CModel 检查',
    'not_planned': '未另设版本',
    'device_fault': '板卡试跑故障',
    'blocked_device_quarantine': 'CModel 通过；设备隔离未计时',
    'blocked_device_safety': '安全排除；保留既有验证证据',
    'skipped_device_safety': '安全排除，未追加 CModel',
    'cmodel_passed_no_timing': 'CModel 通过；待计时',
}
DESCRIPTIONS = {
    'matmul':
        '新表达将完整 K 的 A tile 常驻，并在外层固定 M tile、内层扫描 N，减少 A 重读和 K 分块累加。双方都补充完整 K 候选，配置由各自实测选择。',
    'rmsnorm':
        '新表达保留输入与平方结果，原地更新 FP32 统计量和归一化值；另测共享 gamma 只加载一次并通过精确外积广播，保持 gamma、epsilon 和输出语义。',
    'rmsnorm-splitk':
        '对本轮能放入 local memory 的完整行，另测单次读取的常驻表达，替代两遍宽度扫描。它是独立算法候选；严格消融仍使用原 split-K 与 PPL 流程的相同宽度 tile。',
    'swiglu':
        '新表达在 FP32 中先计算 gate × up，再直接除以 1 + exp(-gate)，省去全 1 buffer 与独立 reciprocal/multiply。低精度 PPL 流程移植版另列。',
    'rope':
        '新表达把 row/pair 展平到 NPU 的 C 维、把偶/奇分量放在 W=0/1，批量执行原 FP32 公式，去掉逐 pair 循环；输入输出只是连续内存的等价视图，无需主机重排。早期直接 DMA reshape 的失败探针单独保留，不上板。',
    'flashattn':
        '新表达保留 FP32 online 状态，省去 non-causal 的零 mask 读取与加法，跳过 causal 的整块未来 KV，并省去首块的 rescale。循环状态显式初始化，避免当前内存规划把跨迭代值当成临时量。'
        'S=8192 的 FP16 TPUKernel 只复测已验证的 BM64/BN256 serial 配置；其余长序列新配置安全排除，不能把这一项解释为充分搜索后的最优。',
    'llama-mlp':
        '新表达将 gate/up 的局部权重合并为一次 GEMM，再切分结果；保留 PPL 的 cast 位置和融合边界。原始权重为 TN，移植版与新表达为 NN，离线权重转置不计入 kernel 时间。',
    'paged-attention':
        '只比较 read-only paged GQA decode。PPL 流程移植版仅对齐 online-softmax，gather 粒度未完全对齐，不能计入严格算法消融。',
    'gptq-w4a16':
        '当前 SDK/芯片的可执行 DQ2 通路不满足要求；不向板卡提交断言或固件诊断桩。',
    'pipeline-add':
        '流水机制对照；数学算法一致，不重复制造算法移植版。',
    'pipeline-matmul':
        '流水机制对照；数学算法一致。RV 当前公共接口不提供该 demo 所用的手动 command group。',
}


def time_text(row):
    return f"{row['p50_us']:.3f}" if row and row.get('p50_us') is not None else '—'


def algorithm_label(row):
    if row['implementation'] == 'tilelang-matched' and row['op'] == 'paged-attention':
        return '局部对齐 TL（gather 粒度不同）'
    if row['implementation'] != 'tilelang-optimized':
        return LABELS[row['implementation']]
    expression = row['config'].get('expression')
    if expression == 'pair_channels':
        return 'FP32 pair 通道向量化'
    if expression == 'shared_gamma':
        return '常驻行 + 共享 gamma 广播'
    return {'matmul': '完整 K 的 A 常驻', 'rmsnorm': '常驻行 + 原地统计',
            'rmsnorm-splitk': '整行常驻', 'swiglu': 'FP32 direct division',
            'llama-mlp': 'gate/up 合并 GEMM', 'flashattn': '首块简化 + causal 剪枝'}.get(
                row['op'], '新优化表达 TL')


def config_text(config):
    return ', '.join(f'{key}={value}' for key, value in config.items())


def short_reason(row):
    reason = row.get('reasons', '')
    if reason.startswith('S>=8192 RV attention comparison is excluded'):
        return '长序列 RV Attention 因既有 Fault 安全排除；保留已有完整验证证据，不再加载其他 tile 或算法变体。'
    if reason.startswith('S>=8192 TPUKernel attention: no new tile'):
        return 'TPUKernel 长序列因 pipeline Fault 收紧范围；只复测 4 个已验证 FP16 serial 程序，本候选安全排除，原数值证据保留。'
    if not reason:
        return ('没有另设算法表达，沿用实际通过的版本' if row['status'] == 'not_planned'
                else '等待验证或计时' if row['status'] == 'pending' else '—')
    parts = reason.splitlines()
    line = parts[-1] if reason.startswith('Traceback') else parts[0]
    return line.replace('|', '/').replace('`', '')[:220]


def write_status(output, document):
    output, document = Path(output).resolve(), Path(document).resolve()
    rows = json.loads((output / 'all_trials.json').read_text())
    comparisons = json.loads((output / 'comparisons.json').read_text())
    coverage = json.loads((output / 'coverage.json').read_text())
    state_file = output / 'study_status.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['case'], row['model'], row['implementation']].append(row)
    comparison = {(row['case'], row['model']): row for row in comparisons}
    cm, hw = (Counter(row[field] for row in rows) for field in ('cmodel_status', 'pcie_status'))
    valid = sum(row['cmodel_status'] == row['pcie_status'] == 'passed' for row in rows)
    final = sum(
        row['timing_stage'] == 'final' and row['pcie_status'] == 'passed' and row['samples'] == 100
        for row in rows)
    controls = {}
    for stage, filename in (('搜索', 'ablation_fixed_tiles_search.json'),
                            ('100 样本复测', 'ablation_fixed_tiles.json')):
        path = output / filename
        for row in json.loads(path.read_text()) if path.exists() else []:
            if row.get('original_us') is not None and row.get('matched_us') is not None:
                controls[row['case'], row['model']] = (stage, row)
    relative = Path(os.path.relpath(output, document.parent)).as_posix()
    detail_dir = document.parent / 'operators'
    detail_dir.mkdir(exist_ok=True)
    # Keep the figures needed to read the report beside the versioned docs.
    # Large native modules, fixtures and full logs stay under artifacts/.
    figure_dir = document.parent / 'figures'
    figure_dir.mkdir(exist_ok=True)
    for figure in sorted((output / 'figures/operators').glob('*.png')):
        shutil.copy2(figure, figure_dir / figure.name)
    lines = [
        '# 算子性能与算法消融', '', '本报告回答两个问题：在相同完整问题上，哪种 TileLang 算法表达最快，与 PPL 相比如何；'
        '固定 tile 和流水配置后，改用 PPL 的算法流程会怎样影响 TileLang 的耗时。', '',
        f"更新：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}。"
    ]
    if state.get('state') != 'complete':
        lines += ['', '**测试尚未全部结束。** ' + state.get('reason', '测试进行中。')]
    elif state.get('reason'):
        lines += ['', '**本轮测试已结束；安全排除与失败项不补时间。** ' + state['reason']]
    lines += [
        '', f'57 组形状/语义组合乘 FP16/BF16，共 {len(cases())} 个完整问题；每个问题分别测试 RV 和 TPUKernel。'
        f"当前有 {cm['passed']} 个候选通过完整 cmodel，{valid} 个有效板卡计时记录，"
        f'{final} 个候选完成 100 样本复测。候选数不是形状数；同一形状会测试多个配置。', '',
        '所有耗时单位均为 µs，统计量为 P50。计时范围是常驻 module/buffer 后的 host launch + stream synchronize；'
        '编译、分配、传输、CPU oracle 不计入。cmodel 耗时不作为性能。', '',
        '“优选 TL”是在原始、PPL 流程移植和新增优化表达中选择实际通过且最快的一项；它可能仍是原始实现。'
        '各版本分别调 tile，因此主比较包含配置选择的收益。严格算法消融另用相同实际 tile/流水配置，展示两边的原始耗时。'
        '速度比为前者耗时除以后者耗时，大于 1 表示后者更快。', '', '## 逐算子结果', '', '每个链接包含一张图、逐形状原始耗时、算法选择、相同 tile 消融及跳过原因。'
        '“双方已测形状数”指该后端下 PPL 和至少一种 TL 均通过正确性并取得计时的完整形状数量；FP16/BF16 分别计数。'
        '速度比按共同已测形状取几何平均，分别列 FP16 / BF16；大于 1 表示优选 TL 更快。'
        '结果只代表这些算子形状，不能解释为整模型加速。', '',
        '| 算子 | RV 已测 / 总数 | RV PPL / 优选 TL（FP16 / BF16） | TPUKernel 已测 / 总数 | TPUKernel PPL / 优选 TL（FP16 / BF16） | 详细数据与图 |',
        '| --- | ---: | ---: | ---: | ---: | --- |'
    ]
    for op in OPERATOR_ORDER:
        op_cases = sorted([case for case in cases() if case['op'] == op], key=case_sort_key)
        measured = {
            model:
                sum(
                    bool((item := comparison.get((
                        case['id'], model))) and item['ppl'] and item.get('best_algorithm'))
                    for case in op_cases) for model in ('rv', 'tpukernel')
        }
        def ratios(model, op_cases=op_cases):
            displayed = []
            for dtype in ('float16', 'bfloat16'):
                values = [item['ppl_over_best_algorithm'] for case in op_cases
                          if case['dtype'] == dtype and
                          (item := comparison.get((case['id'], model))) and
                          item.get('ppl_over_best_algorithm') is not None]
                displayed.append(f'{math.exp(sum(map(math.log, values)) / len(values)):.3f}'
                                 if values else '—')
            return ' / '.join(displayed)

        lines.append(f"| {op} | {measured['rv']} / {len(op_cases)} | {ratios('rv')} | "
                     f"{measured['tpukernel']} / {len(op_cases)} | {ratios('tpukernel')} | "
                     f"[数据与图](operators/{op}.md) |")
        subrelative = Path(os.path.relpath(output, detail_dir)).as_posix()
        detail = [
            f'# {op}', '',
            DESCRIPTIONS.get(op, '二元逐元素算法一致，复用同算法测量；按已测 tile 选配置。'), '', '## 各版本分别调优后的耗时', '',
            '单位 µs，均为 P50；— 表示没有有效板卡计时，不能读作零耗时或 1 倍。'
            '优选 TL 的算法名及配置列在后表。搜索结果与 100 样本复测通过“阶段”区分。', '',
            f'![{op} 性能与消融](../figures/{op}.png)', '',
            '| 完整形状 | dtype | 后端 | PPL | 原始 TL | PPL 算法 TL | 新优化表达 TL | 优选 TL | PPL / 优选 TL | 阶段 |',
            '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |'
        ]
        selected_rows = []
        for case in op_cases:
            for model in ('rv', 'tpukernel'):
                item = comparison.get((case['id'], model), {})
                best = item.get('best_algorithm')
                baseline = item.get('ppl')
                measured_versions = [item[key] for key in ('ppl', 'tilelang', 'matched', 'optimized')
                                     if item.get(key)]
                stage = ('无有效计时' if not measured_versions else
                         '100 样本复测（有效项）' if all(
                             row['timing_stage'] == 'final' and row['samples'] == 100
                             for row in measured_versions) else '含搜索结果，待复测')
                ratio = f"{item['ppl_over_best_algorithm']:.3f}" if item.get(
                    'ppl_over_best_algorithm') is not None else '—'
                shape = '×'.join(map(str,
                                     case['shape'])) + (' causal' if case.get('causal') else '')
                detail.append(
                    f"| {shape} | {case['dtype']} | {model} | {time_text(baseline)} | "
                    f"{time_text(item.get('tilelang'))} | {time_text(item.get('matched'))} | "
                    f"{time_text(item.get('optimized'))} | {time_text(best)} | {ratio} | {stage} |")
                if best:
                    selected_rows.append((shape, case['dtype'], model, best))
        detail += [
            '', '### 优选算法及实际配置', '', '| 完整形状 | dtype | 后端 | 实际选择 | tile / 流水配置 |',
            '| --- | --- | --- | --- | --- |'
        ]
        for shape, dtype, model, best in selected_rows:
            detail.append(f"| {shape} | {dtype} | {model} | {algorithm_label(best)} | "
                          f"{config_text(best['effective_config'])} |")
        detail += [
            '', '## 固定 tile 的算法消融', '', '这部分只比较原始 TL 和 PPL 算法移植版，两个版本使用相同实际 tile/流水配置。'
            '不能将前表各自调优的时间直接当成固定 tile 消融。', '',
            '| 完整形状 | dtype | 后端 | 原始 TL µs | PPL 算法 TL µs | 移植后加速比 | 阶段 | 实际配置 |',
            '| --- | --- | --- | ---: | ---: | ---: | --- | --- |'
        ]
        controlled = 0
        identity = op.startswith('elementwise-') or op in ('matmul', 'pipeline-add',
                                                           'pipeline-matmul')
        for case in op_cases:
            for model in ('rv', 'tpukernel'):
                selected = controls.get((case['id'], model))
                if selected is None or identity or op == 'paged-attention':
                    continue
                stage, item = selected
                shape = '×'.join(map(str,
                                     case['shape'])) + (' causal' if case.get('causal') else '')
                detail.append(
                    f"| {shape} | {case['dtype']} | {model} | {item['original_us']:.3f} | "
                    f"{item['matched_us']:.3f} | {item['original_over_matched']:.3f} | "
                    f"{stage} | {config_text(item['config'])} |")
                controlled += 1
        if not controlled:
            note = ('原始算法已一致，复用同一测量，不构造重复消融。' if identity else '本项尚无通过正确性且实际 tile 一致的严格消融记录。')
            if op == 'paged-attention':
                note = '分页 gather 粒度无法完全对齐，移植版为局部流程对齐，排除严格算法消融。'
            if op == 'rmsnorm':
                note = ('原始 regular TL 使用完整宽度 4096；PPL 样例强制至少两次扫描、实际宽度 tile 至多 2048，'
                        '没有相同实际 tile。前表保留双方原始耗时；严格控制见 rmsnorm-splitk。')
            detail += ['', note]
        detail += [
            '', '## 正确性、支持范围与跳过', '', '每个候选完整 cmodel 通过后才允许上板；失败配置保留证据并跳过，不放宽数值门槛。'
            '一个配置失败不等于整个算子不支持。完整逐候选原因见 '
            f'[all_trials.csv]({subrelative}/all_trials.csv)。', '',
            '| 完整问题 | 后端 | 实现 | cmodel 通过 / 候选数 | 有效板卡配置数 | 当前状态 | 跳过或缺失原因 |',
            '| --- | --- | --- | ---: | ---: | --- | --- |'
        ]
        for row in coverage:
            if row['op'] != op:
                continue
            detail.append(
                f"| {row['case']} | {row['model']} | {LABELS[row['implementation']]} | "
                f"{row['cmodel_passed']} / {row['candidates']} | {row['measured']} | "
                f"{STATUS_LABELS.get(row['status'], row['status'])} | {short_reason(row)} |"
            )
        detail += [
            '', '[来源、算法差别与能力限制](../02_source_mapping.md)；'
            '[板卡故障归因及修复验证](../04_device_incident.md)。', ''
        ]
        (detail_dir / f'{op}.md').write_text('\n'.join(detail))
    lines.append('| moe | 未上板 | — | 未上板 | — | [能力诊断图](operators/moe.md) |')
    (detail_dir / 'moe.md').write_text(
        '# MoE\n\n当前 BM1690E target 在 demo 依赖的 TopK lowering 中拒绝 '
        '`tpu_hau_sort_natural_index`。RV/TPUKernel 均在编译探针阶段跳过，没有完整算子板卡耗时。'
        '这不表示所有 expert GEMM 都不支持。\n\n'
        '![MoE 能力诊断](../figures/moe.png)\n\n'
        '[实现与目标限制](../02_source_mapping.md)。\n')
    lines += [
        '', '## 数据、方法与边界', '',
        f'- [逐候选配置、耗时与失败原因]({relative}/all_trials.csv)', f'- [最佳配置原始记录]({relative}/best.json)',
        f'- [逐形状覆盖]({relative}/coverage.csv)',
        f'- [固定 tile 消融的完整复测数据]({relative}/ablation_fixed_tiles.csv)；'
        f'[搜索阶段数据]({relative}/ablation_fixed_tiles_search.csv)',
        '- [形状、正确性与计时协议](01_methodology.md)', '- [来源、算法移植边界与不支持原因](02_source_mapping.md)',
        '- [故障归因、同步修复与恢复验证](04_device_incident.md)', '', '数学原语近似、归约树和指令选择仍由后端决定。算法移植不等于生成指令逐条相同。'
        '没有有效时间的结果不补值，速度比小于 1 的结果不截断。', '', '| 阶段 | 状态 | 候选数 |', '| --- | --- | ---: |'
    ]
    if (output / 'staged-long-controller.json').exists():
        index = lines.index('## 数据、方法与边界') + 2
        final_audit = output / 'known-long-final-audit.json'
        label = '分阶段诊断与故障记录（原队列已停止）' if final_audit.exists() else '长序列分阶段进度'
        lines[index:index] = [
            f'- [{label}]({relative}/staged-long-controller.json)；'
            f'[完整 CModel 队列进度]({relative}/cmodel-handoff.json)。S=4096 为上板前诊断，不计入性能问题。'
        ]
        if final_audit.exists():
            lines.insert(index + 1, f'- [已验证长序列的 100 样本复测与模块一致性]({relative}/known-long-final-audit.json)')
    for name, counts in (('cmodel', cm), ('PCIe', hw)):
        for status, count in sorted(counts.items()):
            lines.append(f'| {name} | {STATUS_LABELS.get(status, status)} | {count} |')
    document.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--document', type=Path, required=True)
    args = parser.parse_args()
    write_status(args.output, args.document)


if __name__ == '__main__':
    main()
