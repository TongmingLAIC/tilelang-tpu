"""Coverage and paired summaries, including problems with no usable timing."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from .workloads import cases, case_sort_key, OPERATOR_ORDER


def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def write_table(path, rows):
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with path.with_suffix(".csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def summarize(output):
    output = Path(output).resolve()
    from .report import report

    report(output)
    rows = json.loads((output / "all_trials.json").read_text())
    comparisons = json.loads((output / "comparisons.json").read_text())
    state_path = output / 'study_status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case"], row["model"], row["implementation"]].append(row)

    coverage = []
    for case in sorted(cases(), key=case_sort_key):
        identity = case["op"].startswith("elementwise-") or case["op"] in ("matmul", "pipeline-add",
                                                                           "pipeline-matmul")
        for model in ("rv", "tpukernel"):
            for implementation in ("ppl", "tilelang", "tilelang-matched", "tilelang-optimized"):
                shared = identity and implementation == "tilelang-matched"
                candidates = grouped[case["id"], model, "tilelang" if shared else implementation]
                statuses = Counter(row["cmodel_status"] for row in candidates)
                measured = [
                    row for row in candidates
                    if row["cmodel_status"] == "passed" and row["pcie_status"] == "passed"
                ]
                device_faults = sum(row["pcie_status"] == "device_fault" for row in candidates)
                pending = sum(statuses[status] for status in ("not_run", "queued"))
                if measured:
                    status = "measured"
                elif device_faults:
                    status = "device_fault"
                elif not candidates:
                    status = "not_planned"
                elif statuses['unsupported'] == len(candidates):
                    status = 'unsupported'
                elif any(row.get('safety_excluded') for row in candidates):
                    status = 'blocked_device_safety'
                elif statuses["passed"]:
                    status = ('blocked_device_quarantine' if state.get('hardware_available') is False
                              else 'cmodel_passed_no_timing')
                elif pending:
                    status = "pending"
                elif statuses["unsupported"] == len(candidates):
                    status = "unsupported"
                else:
                    status = "failed"
                coverage.append(
                    dict(
                        case=case["id"],
                        op=case["op"],
                        dtype=case["dtype"],
                        model=model,
                        implementation=implementation,
                        status=status,
                        identity_of="tilelang" if shared else "",
                        candidates=len(candidates),
                        cmodel_passed=statuses["passed"],
                        measured=len(measured),
                        device_fault_configs=device_faults,
                        unsupported=statuses["unsupported"],
                        pending=pending,
                        numerical_mismatch=statuses["numerical_mismatch"],
                        timeout=statuses["timeout"],
                        safety_skipped=statuses['skipped_device_safety'],
                        other_failure=sum(count for name, count in statuses.items()
                                          if name not in ("passed", "unsupported", "not_run",
                                                          "queued", "numerical_mismatch",
                                                          "timeout", "skipped_device_safety")),
                        reasons=" | ".join(sorted({row["reason"] for row in candidates if row["reason"]}))
                        or (state.get('reason', '') if status == 'blocked_device_quarantine' else ''),
                    ))
    write_table(output / "coverage", coverage)

    families = defaultdict(list)
    for item in comparisons:
        ppl, tilelang = item["ppl"], item["tilelang"]
        if not ppl or not tilelang:
            continue
        # Mechanism diagnostics stay separate; no weighting by config count.
        families[tilelang["op"], item["model"], tilelang["dtype"]].append(item)
    family_rows = []

    def family_key(item):
        op, model, dtype = item[0]
        return (OPERATOR_ORDER.index(op), 0 if model == "rv" else 1, 0 if dtype == "float16" else 1)

    for (op, model, dtype), items in sorted(families.items(), key=family_key):
        ratios = [item["ppl_over_tilelang"] for item in items]
        family_rows.append(
            dict(
                op=op,
                model=model,
                dtype=dtype,
                paired_shapes=len(items),
                final_pairs=sum(item["ppl"]["timing_stage"] == "final" and
                                item["tilelang"]["timing_stage"] == "final" for item in items),
                ppl_over_tilelang_geomean=geometric_mean(ratios),
                ppl_over_tilelang_min=min(ratios),
                ppl_over_tilelang_max=max(ratios),
                tilelang_faster_shapes=sum(value > 1 for value in ratios),
                ppl_over_best_algorithm_geomean=geometric_mean(
                    [item['ppl_over_best_algorithm'] for item in items]),
            ))
    write_table(output / "family_summary", family_rows)

    controls_path = output / "ablation_fixed_tiles.json"
    controls = json.loads(controls_path.read_text()) if controls_path.exists() else []
    case_info = {case["id"]: case for case in cases()}
    ablations = defaultdict(list)
    fixed = defaultdict(list)
    partial = []
    for item in comparisons:
        if item["matched_is_identical_algorithm"] or not item["tilelang"] or not item["matched"]:
            continue
        if item.get("matched_is_partial_algorithm"):
            partial.append(
                dict(
                    case=item["case"],
                    model=item["model"],
                    original_us=item["tilelang"]["p50_us"],
                    partial_port_us=item["matched"]["p50_us"],
                    reason="Online attention aligns, but exact per-head gather views are unavailable through compiler-owned TPU descriptors; excluded from strict algorithm ablation."
                ))
            continue
        key = item["tilelang"]["op"], item["model"], item["tilelang"]["dtype"]
        ablations[key].append(item["tilelang"]["p50_us"] / item["matched"]["p50_us"])
    for item in controls:
        if item["identical_algorithm"] or item["original_over_matched"] is None:
            continue
        case = case_info[item["case"]]
        fixed[case["op"], item["model"], case["dtype"]].append(item["original_over_matched"])
    ablation_rows = []
    for (op, model, dtype), ratios in sorted(ablations.items(), key=family_key):
        controlled = fixed[op, model, dtype]
        ablation_rows.append(
            dict(
                op=op,
                model=model,
                dtype=dtype,
                independently_tuned_pairs=len(ratios),
                original_over_matched_geomean=geometric_mean(ratios),
                fixed_tile_pairs=len(controlled),
                fixed_tile_original_over_matched_geomean=geometric_mean(controlled)
                if controlled else None,
            ))
    write_table(output / "ablation_summary", ablation_rows)
    write_table(output / "partial_alignment", partial)

    lines = [
        "# 覆盖与共同实测形状汇总", "", "速度比为 PPL P50 / TileLang P50，大于 1 表示 TileLang 更快。"
        "几何平均只覆盖同后端、同 dtype、两边均通过的完整问题；不能解释为整模型加速。"
        "完成复测的形状数少于双方已测形状数时，该行仍含搜索阶段计时。", "",
        "| 算子 | 后端 | dtype | 双方已测形状数 | 其中完成 100 样本复测 | PPL / 原 TL | 范围 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |"
    ]
    for row in family_rows:
        lines.append(f"| {row['op']} | {row['model']} | {row['dtype']} | "
                     f"{row['paired_shapes']} | {row['final_pairs']} | "
                     f"{row['ppl_over_tilelang_geomean']:.3f} | "
                     f"{row['ppl_over_tilelang_min']:.3f}–{row['ppl_over_tilelang_max']:.3f} |")
    lines.extend([
        "", "逐形状支持情况见 `coverage.csv`。identity_of 指向共享的实际测量，"
        "不把相同算法的复用条目计为额外候选。measured 只表示至少一个配置有效，"
        "同组其他配置的错误仍在各错误列和 all_trials.csv 中保留。", "", "| 覆盖状态 | case / backend / implementation 条目 |",
        "| --- | ---: |"
    ])
    for status, count in sorted(Counter(row["status"] for row in coverage).items()):
        lines.append(f"| {status} | {count} |")
    if ablation_rows:
        lines.extend([
            "", "## 算法消融", "", "原 TL / 同算法 TL 大于 1 表示移植 PPL 算法后更快。独立调优列也包含 tile 选择的影响；"
            "固定 tile 列使用实际 tile 相同的控制实验。指数近似、归约树和指令选择仍由后端决定。"
            "identity 算法不计作额外消融。", "",
            "| 算子 | 后端 | dtype | 独立调优已测形状数 | 原 TL / 同算法 TL | 同 tile 已测形状数 | 同 tile 比值 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |"
        ])
        for row in ablation_rows:
            controlled = row["fixed_tile_original_over_matched_geomean"]
            displayed = f"{controlled:.3f}" if controlled is not None else "—"
            lines.append(
                f"| {row['op']} | {row['model']} | {row['dtype']} | "
                f"{row['independently_tuned_pairs']} | {row['original_over_matched_geomean']:.3f} | "
                f"{row['fixed_tile_pairs']} | {displayed} |")
    if partial:
        lines.extend([
            "", "分页注意力的 online-softmax 局部移植结果另见 `partial_alignment.json`；"
            "public gather 无法与 PPL 的 head sub-view 对齐，未计入上述严格算法消融。"
        ])
    (output / "ANALYSIS.md").write_text("\n".join(lines) + "\n")
    return coverage, family_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)


if __name__ == "__main__":
    main()
