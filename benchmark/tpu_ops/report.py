"""Aggregate only measured, numerically passing, paired trials."""
import csv
import json
import os
from pathlib import Path
from collections import Counter


def write_json(path, value):
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def report(output):
    from .workloads import effective_config, case_sort_key
    output = Path(output)
    exclusions_file = output / 'safety_exclusions.json'
    exclusions = json.loads(exclusions_file.read_text()) if exclusions_file.exists() else {}
    rows = []
    case_definitions = {}
    for path in sorted(output.glob("*/*/*/*/trial.json")):
        spec = json.loads(path.read_text())
        case_definitions[spec["case"]["id"]] = spec["case"]
        trial = path.parent
        exclusion = exclusions.get(str(trial.resolve()), {})
        cm_path = trial / "cmodel/result.json"
        hw_path = trial / "pcie/result.json"
        cm = json.loads(cm_path.read_text()) if cm_path.exists() else {"status": "not_run"}
        if not cm_path.exists() and exclusion.get('skip_cmodel'):
            cm = dict(status='skipped_device_safety', reason=exclusion['reason'])
        hw = json.loads(hw_path.read_text()) if hw_path.exists() else {"status": "not_run"}
        canary_path = trial / 'pcie-canary/result.json'
        if not hw_path.exists() and canary_path.exists():
            canary = json.loads(canary_path.read_text())
            if canary['status'] != 'passed':
                hw = dict(canary, reason=canary.get('reason') or
                          f"board numerical canary {canary['status']}; repeated timing skipped")
        search_p50 = hw.get("timing", {}).get("p50")
        final_path = trial / "pcie-final/result.json"
        timing_stage = "search"
        if final_path.exists():
            hw = json.loads(final_path.read_text())
            timing_stage = "final"
        device_path = trial / "pcie-profile/device_timing.json"
        device = json.loads(device_path.read_text()) if device_path.exists() else {}
        reason = hw.get('reason') or cm.get('reason') or (
            exclusion.get('reason', '') if hw['status'] == 'not_run' else '')
        if not reason and cm['status'] == 'numerical_mismatch':
            check = next((value for value in cm.get('checks', []) if not value.get('passed')), {})
            reason = (f"cmodel numerical_mismatch: {check.get('mismatched')} / "
                      f"{check.get('elements')} outputs outside tolerance; "
                      f"max_abs={check.get('max_abs')}")
        if not reason and cm['status'] == 'timeout':
            reason = 'complete-shape cmodel exceeded its recorded resource budget; board skipped'
        rows.append(
            dict(
                case=spec["case"]["id"],
                op=spec["case"]["op"],
                shape=spec["case"]["shape"],
                dtype=spec["case"]["dtype"],
                model=spec["model"],
                implementation=spec["implementation"],
                config=spec["config"],
                effective_config=effective_config(spec["case"], spec["config"],
                                                  spec["implementation"]),
                cmodel_status=cm["status"],
                pcie_status=hw["status"],
                timing_stage=timing_stage,
                samples=hw.get("timing", {}).get("samples"),
                search_p50_us=search_p50,
                device_command_span_us=device.get("device_command_span_us"),
                p50_us=hw.get("timing", {}).get("p50"),
                p90_us=hw.get("timing", {}).get("p90"),
                reason=reason,
                safety_excluded=bool(exclusion and hw['status'] != 'passed'),
                safety_exclusion_reason=exclusion.get('reason', ''),
                path=str(trial)))
    rows.sort(key=lambda row: (case_sort_key(case_definitions[row["case"]]), 0 if row[
        "model"] == "rv" else 1, ("ppl", "tilelang", "tilelang-matched", "tilelang-optimized").
                               index(row["implementation"]), tuple(sorted(row["config"].items()))))
    write_json(output / 'all_trials.json', rows)
    if rows:
        with (output / "all_trials.csv").open("w") as file:
            w = csv.DictWriter(file, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    best = {}
    finalized_groups = {(r["case"], r["model"], r["implementation"])
                        for r in rows
                        if r["timing_stage"] == "final" and r["pcie_status"] == "passed"}
    for row in rows:
        if row["cmodel_status"] != "passed" or row["pcie_status"] != "passed" or row[
                "p50_us"] is None:
            continue
        key = (row["case"], row["model"], row["implementation"])
        if key in finalized_groups and row["timing_stage"] != "final":
            continue
        if key not in best or row["p50_us"] < best[key]["p50_us"]:
            best[key] = row
    comparisons = []
    for case, model in sorted({(key[0], key[1]) for key in best},
                              key=lambda item: (case_sort_key(case_definitions[item[0]]), 0
                                                if item[1] == "rv" else 1)):
        ppl = best.get((case, model, "ppl"))
        tl = best.get((case, model, "tilelang"))
        ab = best.get((case, model, "tilelang-matched"))
        optimized = best.get((case, model, "tilelang-optimized"))
        identity = bool(
            tl and (tl["op"].startswith("elementwise-") or
                    tl["op"] in ("matmul", "pipeline-add", "pipeline-matmul")))
        if identity and ab is None:
            ab = tl
        algorithms = [row for row in (tl, ab, optimized) if row is not None]
        selected = min(algorithms, key=lambda row: row['p50_us']) if algorithms else None
        comparisons.append(
            dict(
                case=case,
                model=model,
                ppl=ppl,
                tilelang=tl,
                matched=ab,
                optimized=optimized,
                best_algorithm=selected,
                matched_is_identical_algorithm=identity,
                matched_is_partial_algorithm=bool(ab and ab["op"] == "paged-attention"),
                ppl_over_tilelang=ppl["p50_us"] / tl["p50_us"] if ppl and tl else None,
                ppl_over_best_algorithm=ppl['p50_us'] /
                selected['p50_us'] if ppl and selected else None,
                matched_over_original=ab["p50_us"] / tl["p50_us"] if ab and tl else None))
    write_json(output / 'best.json', list(best.values()))
    write_json(output / 'comparisons.json', comparisons)
    controlled = []
    if (output / "controls.json").exists():
        for item in json.loads((output / "controls.json").read_text()):

            def final_time(key, item=item):
                if not item.get(key):
                    return None
                path = Path(item[key]) / "pcie-final/result.json"
                value = json.loads(path.read_text()) if path.exists() else {}
                return value.get("timing",
                                 {}).get("p50") if value.get("status") == "passed" else None

            orig, matched, ppl = final_time("original"), final_time("matched"), final_time("ppl")
            controlled.append(
                dict(
                    case=item["case"],
                    model=item["model"],
                    config=item["config"],
                    identical_algorithm=item.get("identical_algorithm", False),
                    original_us=orig,
                    matched_us=matched,
                    ppl_us=ppl,
                    original_over_matched=orig / matched if orig is not None and matched else None,
                    ppl_over_matched=ppl / matched if ppl is not None and matched else None))
    write_json(output / 'ablation_fixed_tiles.json', controlled)
    if controlled:
        with (output / "ablation_fixed_tiles.csv").open("w") as file:
            w = csv.DictWriter(file, fieldnames=controlled[0].keys())
            w.writeheader()
            w.writerows(controlled)
    statuses = Counter((r["cmodel_status"], r["pcie_status"]) for r in rows)
    final_count = sum(r["timing_stage"] == "final" and r["pcie_status"] == "passed" for r in rows)
    lines = [
        "# 算子比较实测结果", "", f"候选数：{len(rows)}；有板卡计时的最优实现条目：{len(best)}。", "",
        f"计时为驻留 module/buffer 后的 host launch + stream synchronize；不含编译、初始化和 H2D/D2H。单位 µs。每个候选先完成完整形状 cmodel 数值检查；具体 seed 清单保存在 trial.json。搜索每个 seed 计时 10 次；最终阶段计划对每组前两名分别取 100 样本，目前已完成 {final_count} 个候选。阶段和实际样本数保存在 CSV。",
        "", "速度比为 PPL / TileLang，保留小于 1 的结果。双方分别在已执行候选中选最优，不宣称全搜索空间最优。", "",
        "| Case | 后端 | PPL µs | 原 TL µs | PPL 算法 TL µs | 新优化表达 µs | 优选 TL µs | 优选算法 | PPL / 优选 TL |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |"
    ]
    state_path = output / "study_status.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("state") != "complete":
            lines[2:2] = ["**本轮尚未完成：** " + state.get("reason", state["state"]), ""]

    def fmt(value):
        return f"{value:.3f}" if value is not None else "—"

    for item in comparisons:
        displayed_matched = fmt((item['matched'] or {}).get('p50_us'))
        if item['matched_is_partial_algorithm']:
            displayed_matched = 'partial: ' + displayed_matched
        lines.append(
            f"| {item['case']} | {item['model']} | {fmt((item['ppl'] or {}).get('p50_us'))} | {fmt((item['tilelang'] or {}).get('p50_us'))} | {displayed_matched} | {fmt((item['optimized'] or {}).get('p50_us'))} | {fmt((item['best_algorithm'] or {}).get('p50_us'))} | {(item['best_algorithm'] or {}).get('implementation', '—')} | {fmt(item['ppl_over_best_algorithm'])} |"
        )
    lines += [
        '',
        'partial 表示分页注意力只对齐 online-softmax，public gather 无法表达 PPL 的相同 head sub-view；这项不计入严格算法消融汇总。'
    ]
    lines += ["", "| CModel | PCIe | 候选数 |", "| --- | --- | ---: |"]
    for (cm, hw), n in sorted(statuses.items()):
        lines.append(f"| {cm} | {hw} | {n} |")
    if any(item["original_us"] is not None and item["matched_us"] is not None
           for item in controlled):
        lines += [
            "", "## 相同 tile / 流水配置的控制实验", "",
            "优先选取三方共有的配置，各以 100 样本复测。identity 表示原 TileLang 已使用该算法，两个 TileLang 列指向同一次测量。regular RMSNorm 的完整行算法没有与宽度分块算法相同的 width tile，不能伪造对应控制；split-K 提供相同 tile 的对照。",
            "", "| Case | 后端 | 原 TL | 同算法 TL | PPL | 原 TL / 同算法 TL |",
            "| --- | --- | ---: | ---: | ---: | ---: |"
        ]
        for item in controlled:
            if item["original_us"] is None or item["matched_us"] is None:
                continue
            lines.append(
                f"| {item['case']} | {item['model']} | {fmt(item['original_us'])} | {fmt(item['matched_us'])} | {fmt(item['ppl_us'])} | {'identity' if item['identical_algorithm'] else fmt(item['original_over_matched'])} |"
            )
    if any(r["device_command_span_us"] is not None for r in rows):
        lines += [
            "", "## 独立 PMU 指令跨度", "", "每个胜出配置另录一次，单位 µs；该值不是重复 launch 的分位数。", "",
            "| Case | 后端 | PPL 指令跨度 | TileLang 指令跨度 | PPL / TL |",
            "| --- | --- | ---: | ---: | ---: |"
        ]
        for item in comparisons:
            pp = (item["ppl"] or {}).get("device_command_span_us")
            tt = (item["tilelang"] or {}).get("device_command_span_us")
            if pp is None and tt is None:
                continue
            lines.append(
                f"| {item['case']} | {item['model']} | {fmt(pp)} | {fmt(tt)} | {fmt(pp/tt if pp is not None and tt else None)} |"
            )
    lines += [
        "",
        "完整配置、状态和原因见 `all_trials.csv`；最优配置见 `best.json`；原始日志、二进制输入和数值检查按 case/backend/implementation/config 排列。"
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "trials": len(rows),
                "best": len(best),
                "statuses": {
                    str(k): v for k, v in statuses.items()
                }
            },
            ensure_ascii=False),
        flush=True)


if __name__ == "__main__":
    import sys
    report(sys.argv[1])
