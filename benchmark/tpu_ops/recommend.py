"""Codex ranks an enumerated, capacity-filtered search space.

The model returns configuration IDs only. The trusted runner owns compilation,
cmodel promotion, device locking, and measurement. This follows the executable
feedback loop of Autocomp / KernelAgent without delegating device access.
"""
import argparse
import json
from pathlib import Path
import subprocess
from .workloads import cases, configs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model", help="Optional override; otherwise use the installed Codex model configuration")
    p.add_argument("--effort", choices=("low", "medium", "high", "xhigh"), default="medium")
    p.add_argument("--feedback", type=Path)
    p.add_argument('--op', action='append', default=[], help='Recommend only these operators')
    p.add_argument(
        '--reuse-answer', action='store_true', help='Revalidate the saved model response')
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    keys = {}
    for c in cases():
        if args.op and c['op'] not in args.op:
            continue
        for model in ("rv", "tpukernel"):
            for impl in ("ppl", "tilelang", "tilelang-matched", "tilelang-optimized"):
                cs = configs(c, impl, model)
                if not cs:
                    continue
                # Group by actual configuration list and phase; recommendations
                # are shared only where tile choices and constraints agree.
                family = "elementwise" if c["op"] in ("elementwise-add", "elementwise-sub",
                                                      "elementwise-mul") else c["op"]
                key = json.dumps(
                    [family, "16-bit", model, impl, cs,
                     c.get('phase'),
                     c.get('causal', False)],
                    sort_keys=True)
                if key not in keys:
                    keys[key] = str(len(keys))
                    groups[keys[key]] = dict(
                        op=c["op"],
                        dtype=c["dtype"],
                        backend=model,
                        implementation=impl,
                        shape=c["shape"],
                        candidates=cs)
                groups[keys[key]].setdefault("case_keys", []).append(f"{c['id']}/{model}/{impl}")
    schema = {
        "type": "object",
        "properties": {
            "rankings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group": {
                            "type": "string"
                        },
                        "ids": {
                            "type": "array",
                            "items": {
                                "type": "integer"
                            }
                        },
                        "reason": {
                            "type": "string"
                        }
                    },
                    "required": ["group", "ids", "reason"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["rankings"],
        "additionalProperties": False
    }
    (out / "schema.json").write_text(json.dumps(schema, indent=2))
    request = dict(
        hardware={
            "chip": "SG2260E",
            "lanes": 64,
            "local_bytes_per_lane": 262144,
            "banks_per_lane": 16,
            "eu_bytes": 64,
            "physical_cores": 4,
            "used_cores": 1,
            "clock_MHz": 1000,
            "device_memory_GB": 128
        },
        protocol="Each candidate must pass full-shape cmodel checks before PCIe. Score is median of 100 resident launch+sync measurements. Separate implementation budgets. stages=2 in PPL means compiler-managed enable_pipeline, not a guaranteed numerical stage count. RV TileLang has no parallel command-group lowering. Choose three different tiles where possible; include a pipeline-on versus pipeline-off comparison for PPL when it fits the budget. No invented measurements.",
        groups={
            k: {
                kk: vv for kk, vv in v.items() if kk != 'case_keys'
            } for k, v in groups.items()
        })
    if args.feedback:
        feedback = json.loads(args.feedback.read_text())
        if args.op:
            feedback = [row for row in feedback if row.get('op') in args.op]
        fields = ('case', 'model', 'implementation', 'config', 'cmodel_status', 'pcie_status',
                  'p50_us', 'timing_stage')
        request['measured_feedback'] = [{
            key: row[key] for key in fields if key in row
        } for row in feedback]
    prompt = "Rank legal tile configurations for the TPU experiment below. Return at most 3 candidate IDs for every group, best-first. Reason about DMA reuse, local-memory capacity and command overhead. Do not call tools, execute commands, or edit files. Return only the requested JSON.\n" + json.dumps(
        request, separators=(',', ':'))
    (out / "request.json").write_text(json.dumps(request, indent=2))
    command = [
        "codex", "exec", "--ephemeral", "--sandbox", "read-only", "--json", "--output-schema",
        str(out / "schema.json"), "--output-last-message",
        str(out / "answer.json")
    ]
    if args.model:
        command += ["--model", args.model]
    if args.effort:
        command += ["-c", f'model_reasoning_effort="{args.effort}"']
    command += ["-"]
    (out / "command.json").write_text(json.dumps(command))
    if not args.reuse_answer:
        with (out / "events.jsonl").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
            result = subprocess.run(
                command, input=prompt, text=True, stdout=stdout, stderr=stderr, timeout=600)
        if result.returncode:
            raise RuntimeError(f"Codex recommendation failed; see {out/'stderr.log'}")
    answer = json.loads((out / "answer.json").read_text())
    recommendations = {}
    skips = {}
    observed = set()
    for item in answer["rankings"]:
        key = item["group"]
        ids = item["ids"]
        if key not in groups or key in observed or len(ids) > 3 or len(set(ids)) != len(ids):
            raise ValueError("invalid group/duplicate/empty recommendation")
        if not ids:
            group = groups[key]
            op, impl, backend = group['op'], group['implementation'], group['backend']
            low_exp = (
                backend == 'rv' and impl == 'tilelang-matched' and
                op in ('swiglu', 'flashattn', 'paged-attention'))
            descriptor = (
                op == 'paged-attention' and impl == 'tilelang' and
                group['shape'][0] * 16 * group['shape'][1] > 65535)
            if not (low_exp or descriptor):
                raise ValueError('model declined an unproven group; no unverified skip accepted')
            for case_key in group['case_keys']:
                skips[case_key] = dict(
                    model_reason=item['reason'],
                    verified_capability='RV storage exp lowering'
                    if low_exp else 'public embedding descriptor axis >65535')
        if any(type(i) is not int or not 0 <= i < len(groups[key]["candidates"]) for i in ids):
            raise ValueError("out-of-space candidate")
        observed.add(key)
        for case_key in groups[key]["case_keys"]:
            recommendations[case_key] = ids
    if observed != set(groups):
        raise ValueError("Codex omitted recommendation groups")
    (out / "recommendations.json").write_text(json.dumps(recommendations, indent=2))
    (out / 'skips.json').write_text(json.dumps(skips, indent=2) + '\n')
    spaces = {
        case_key: group["candidates"] for group in groups.values() for case_key in group["case_keys"]
    }
    (out / "spaces.json").write_text(json.dumps(spaces, indent=2))
    print(
        f"Validated rankings for {len(groups)} groups / {len(recommendations)} case/backend/implementation combinations"
    )


if __name__ == "__main__":
    main()
