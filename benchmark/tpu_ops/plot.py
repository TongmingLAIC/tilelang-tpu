"""Export paired performance figures (optional matplotlib dependency)."""

import argparse
import json
import math
from pathlib import Path


def plot_operators(output, plt):
    from .workloads import cases, case_sort_key, OPERATOR_ORDER
    items = json.loads((output / 'comparisons.json').read_text())
    lookup = {(item['case'], item['model']): item for item in items}
    controls = {}
    for filename in ('ablation_fixed_tiles_search.json', 'ablation_fixed_tiles.json'):
        path = output / filename
        for item in json.loads(path.read_text()) if path.exists() else []:
            if item.get('original_us') is not None and item.get('matched_us') is not None:
                controls[item['case'], item['model']] = item
    directory = output / 'figures/operators'
    directory.mkdir(parents=True, exist_ok=True)
    for op in OPERATOR_ORDER:
        definitions = sorted([case for case in cases() if case['op'] == op], key=case_sort_key)
        fig, axes = plt.subplots(3, 2, figsize=(19, 15))

        def bars(axis, labels, values, legends, colors):
            usable = [value for group in values for value in group if value is not None]
            if not usable:
                axis.text(
                    .5,
                    .5,
                    'No valid timing\nSee correctness / support table',
                    ha='center',
                    va='center',
                    transform=axis.transAxes)
                axis.set_xticks([])
                axis.set_yticks([])
                return
            width = .75 / len(values)
            for series, (group, legend, color) in enumerate(zip(values, legends, colors, strict=True)):
                for i, value in enumerate(group):
                    x = i + (series - (len(values) - 1) / 2) * width
                    if value is None:
                        axis.text(x, min(usable) * .65, '—', ha='center', fontsize=7)
                        continue
                    axis.bar(
                        x, value, width=width * .92, color=color, label=legend if i == 0 else None)
                    axis.text(
                        x,
                        value * 1.06,
                        f'{value:.3g}',
                        ha='center',
                        va='bottom',
                        fontsize=6,
                        rotation=90 if len(labels) >= 6 else 0)
            axis.set_yscale('log')
            axis.set_ylim(min(usable) * .5, max(usable) * (6 if len(labels) >= 6 else 3))
            axis.set_ylabel('P50 latency (us, log scale)')
            axis.set_xlim(-.6, len(labels) - .4)
            axis.set_xticks(
                range(len(labels)),
                labels,
                rotation=50 if len(labels) > 3 else 20,
                ha='right',
                fontsize=7)
            axis.grid(axis='y', alpha=.2)
            axis.set_axisbelow(True)
            # Use explicit handles even when the first shape has no result.
            from matplotlib.patches import Patch
            axis.legend(
                handles=[Patch(color=color, label=label) for label, color in zip(legends, colors, strict=True)],
                fontsize=7,
                loc='upper left')

        for col, model in enumerate(('rv', 'tpukernel')):
            for row, dtype in enumerate(('float16', 'bfloat16')):
                selected = [case for case in definitions if case['dtype'] == dtype]
                values = [[], [], []]
                labels = []
                for case in selected:
                    item = lookup.get((case['id'], model), {})
                    labels.append('x'.join(map(str, case['shape'])) +
                                  (' C' if case.get('causal') else ''))
                    for group, key in zip(values, ('ppl', 'tilelang', 'best_algorithm'), strict=True):
                        group.append((item.get(key) or {}).get('p50_us'))
                axis = axes[row, col]
                bars(axis, labels, values,
                     ('PPL: independently tuned', 'Original TL: independently tuned',
                      'Selected TL: best measured algorithm + tile'),
                     ('#64748b', '#2563eb', '#16a34a'))
                axis.set_title(f'{model.upper()} / {dtype}: algorithm + tile selection')
            axis = axes[2, col]
            selected = [(case, controls[case['id'], model])
                        for case in definitions
                        if (case['id'], model) in controls and not controls[case['id'], model].get(
                            'identity', controls[case['id'], model].get('identical_algorithm'))]
            if selected and op != 'paged-attention':
                labels = [('F16 ' if case['dtype'] == 'float16' else 'BF16 ') +
                          'x'.join(map(str, case['shape'])) + (' C' if case.get('causal') else '')
                          for case, _ in selected]
                bars(axis, labels, [[item['original_us'] for _, item in selected],
                                    [item['matched_us'] for _, item in selected]],
                     ('Original TL: same-tile control', 'PPL-algorithm TL: same-tile control'),
                     ('#2563eb', '#d97706'))
            else:
                if op.startswith('elementwise-') or op in (
                        'matmul', 'pipeline-add', 'pipeline-matmul'):
                    reason = 'Original and PPL-flow algorithms are identical.\nThe same measurement is reused.'
                elif op == 'rmsnorm':
                    reason = ('Regular TL uses width 4096; PPL uses width <=2048.\n'
                              'No identical effective tile. See rmsnorm-splitk controls.')
                elif op == 'paged-attention':
                    reason = ('Gather granularity differs: partial algorithm alignment.\n'
                              'Excluded from strict algorithm ablation.')
                elif model == 'rv' and op in ('swiglu', 'flashattn'):
                    reason = ('PPL-flow TL requires storage-dtype exp.\n'
                              'The current RV lowering supports FP32 exp only.')
                elif op == 'gptq-w4a16':
                    reason = 'DQ2 has no executable path on this SDK/target.\nNo board timing or algorithm ablation.'
                else:
                    reason = 'No valid identical-tile algorithm control.\nNo synthetic ablation value is plotted.'
                axis.text(
                    .5,
                    .5, reason,
                    ha='center',
                    va='center',
                    transform=axis.transAxes,
                    fontsize=9)
                axis.set_xticks([])
                axis.set_yticks([])
            axis.set_title(f'{model.upper()}: algorithm ablation with identical effective tiles')
        state_path = output / 'study_status.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        status = {'complete': 'completed measurements',
                  'hardware_blocked': 'board quarantined; available measurements'}.get(
                      state.get('state'), 'measurements in progress')
        fig.suptitle(f'{op} | BM1690E, one core, 1000 MHz | {status}', fontsize=15)
        fig.text(
            .5,
            .012, 'Top rows: independent tile search. Bottom row: same-tile algorithm controls. '
            'C = causal. Lower latency is better.\n'
            'Resident host launch + stream synchronize; compilation and transfers excluded. '
            'Missing / failing cases have no timing value. See tables for search versus final samples.',
            ha='center',
            fontsize=9)
        fig.tight_layout(rect=(0, .055, 1, .96), h_pad=2.5)
        for suffix in ('png', 'svg', 'pdf'):
            fig.savefig(directory / f'{op}.{suffix}', dpi=150)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.axis('off')
    axis.set_title('MoE demo / BM1690E: skipped before board execution', fontsize=14)
    table = axis.table(
        cellText=[['RV', 'TopK lowering rejected', 'No full-kernel timing'],
                  ['TPUKernel', 'TopK lowering rejected', 'No full-kernel timing']],
        colLabels=['Backend', 'Required capability', 'Measurement'], cellLoc='center',
        loc='center', colWidths=[.18, .42, .4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    fig.text(.5, .12, 'The current BM1690E target rejects tpu_hau_sort_natural_index.\n'
             'This limits the demo routing path; it does not imply all expert GEMMs are unsupported.',
             ha='center', fontsize=10)
    fig.tight_layout()
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(directory / f'moe.{suffix}', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    rows = json.loads((output / "family_summary.json").read_text())
    if not rows:
        raise RuntimeError("No measured pairs; run benchmark.tpu_ops.summarize first")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_operators(output, plt)

    from .workloads import OPERATOR_ORDER
    operations = [op for op in OPERATOR_ORDER if any(row["op"] == op for row in rows)]
    lookup = {(row["op"], row["model"], row["dtype"]): row for row in rows}
    finalized = all(row["final_pairs"] == row["paired_shapes"] for row in rows)
    figure, axes = plt.subplots(
        1, 2, figsize=(12, max(6,
                               len(operations) * 0.5)), sharex=True, sharey=True)
    styles = (("float16", "FP16", "#2563eb", -0.13), ("bfloat16", "BF16", "#d97706", 0.13))
    minimum = min(row["ppl_over_tilelang_min"] for row in rows)
    maximum = max(row["ppl_over_tilelang_max"] for row in rows)
    lower = 2**math.floor(math.log2(min(minimum, 1)))
    upper = 2**math.ceil(math.log2(max(maximum, 1)))
    for axis, model in zip(axes, ("rv", "tpukernel"), strict=True):
        for index, operation in enumerate(operations):
            for dtype, label, color, offset in styles:
                row = lookup.get((operation, model, dtype))
                if row is None:
                    continue
                center = row["ppl_over_tilelang_geomean"]
                axis.errorbar(
                    center,
                    index + offset,
                    # A one-shape geometric mean can differ by one ULP from
                    # that same shape's ratio after log/exp round-tripping.
                    xerr=[[max(0.0, center - row["ppl_over_tilelang_min"])],
                          [max(0.0, row["ppl_over_tilelang_max"] - center)]],
                    fmt="o",
                    markersize=5,
                    capsize=3,
                    color=color,
                    label=label if index == 0 else None)
        axis.axvline(1, color="#111827", linewidth=1, linestyle="--")
        axis.set_xscale("log", base=2)
        axis.set_xlim(lower / 1.15, upper * 1.15)
        axis.set_title("RV" if model == "rv" else "TPUKernel")
        axis.set_xlabel("PPL P50 / original TileLang P50 (>1: TileLang faster)")
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].set_yticks(
        range(len(operations)), [
            operation + (" [mechanism]" if operation.startswith("pipeline-") else "")
            for operation in operations
        ])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left")
    figure.suptitle("BM1690E, one TPU core — " +
                    ("100-sample finalists" if finalized else "search in progress"))
    figure.text(
        0.5,
        0.01,
        "Dots: geometric mean across shapes measured by both. Whiskers: shape range, not confidence intervals.\n"
        "Resident kernel launch + stream synchronize; compilation and transfers excluded.",
        ha="center",
        fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(directory / f"paired_performance.{suffix}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
