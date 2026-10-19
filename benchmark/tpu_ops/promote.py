"""Review validated long-attention programs, then run one numerical canary first."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from .command_safety import require_board_source
from .run import ROOT, execute_trial, health, supervised


def inspect_trial(trial):
    """A simulator verdict alone cannot authorize an unbounded device source."""
    spec_file = trial / 'trial.json'
    spec = json.loads(spec_file.read_text())
    result = json.loads((trial / 'cmodel/result.json').read_text())
    if result['status'] != 'passed':
        raise ValueError('only passing complete CModel trials can be promoted')
    if result['trial_sha256'] != hashlib.sha256(spec_file.read_bytes()).hexdigest():
        raise ValueError('validated trial specification has changed')
    source = trial / 'cmodel/kernel.c'
    audit = trial / 'cmodel/command-epoch-audit.json'
    require_board_source(spec, source.read_text(), hashlib.sha256(source.read_bytes()).hexdigest(),
                         json.loads(audit.read_text()) if audit.exists() else None)
    return spec


def canary(trial):
    spec = inspect_trial(trial)
    result_file = trial / 'pcie-canary/result.json'
    if result_file.exists():
        return json.loads(result_file.read_text())['status']
    from tilelang.jit.adapter.tpu_profiling import TPUInstructionProfiler
    env = dict(os.environ, PYTHONPATH=str(ROOT), PPL_PROJECT_ROOT=spec['sdk'],
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    cmd = [sys.executable, '-m', 'benchmark.tpu_ops.worker', '--trial', str(trial),
           '--runtime', 'pcie', '--single-launch']
    if env.get('TILELANG_BENCH_PCIE_CPUS'):
        cmd = ['taskset', '-c', env['TILELANG_BENCH_PCIE_CPUS'], *cmd]
    with TPUInstructionProfiler.exclusive_pcie_device(0):
        health(trial, 'canary-before')
        rc = supervised(cmd, trial / 'canary-launch.log', env, spec['timeout'] + 90,
                        board_result=result_file)
        status = json.loads(result_file.read_text()).get('status') if result_file.exists() else None
        if rc in (124, 125) or status in ('timeout', 'device_fault'):
            TPUInstructionProfiler.fail_closed_pcie_device(0, reason=f'numerical canary stalled: {trial}')
            raise RuntimeError('canary timeout or fault; board quarantined')
        health(trial, 'canary-after', result_path=result_file)
    if not status:
        raise RuntimeError(f'canary produced no verdict: {trial}')
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--minimum-sequence', type=int, default=8192)
    args = p.parse_args()
    output = args.output.resolve()
    execution = json.loads((output / 'execution.json').read_text())
    os.environ['TILELANG_BENCH_PCIE_CPUS'] = execution['pcie_cpus']
    # Snapshot only currently finished full CModel proofs. Concurrent simulator
    # workers can finish later; invoking this command again reviews those then.
    candidates = []
    for proof in sorted(output.glob('flashattn-*/*/*/*/cmodel/result.json')):
        trial = proof.parent.parent
        spec = json.loads((trial / 'trial.json').read_text())
        if spec['case']['shape'][1] < args.minimum_sequence or (trial / 'pcie/result.json').exists():
            continue
        if json.loads(proof.read_text())['status'] != 'passed':
            continue
        inspect_trial(trial)
        candidates.append(trial)
    print(f'Reviewed {len(candidates)} validated long programs; board execution is serial', flush=True)
    for trial in candidates:
        status = canary(trial)
        print(f'Canary: {trial.relative_to(output)}: {status}', flush=True)
        if status != 'passed':
            # The failing canary stays separate from valid performance data.
            print('Skipped repeated timing after failed numerical canary', flush=True)
            continue
        print(f'Search: {trial.relative_to(output)}: {execute_trial(trial, True)}', flush=True)


if __name__ == '__main__':
    main()
