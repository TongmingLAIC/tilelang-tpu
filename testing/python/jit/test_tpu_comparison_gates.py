"""Fail-closed promotion checks; no accelerator or vendor runtime is loaded."""
import json
import hashlib
import sys
import pytest
from types import SimpleNamespace


def test_a_failing_full_shape_cmodel_result_cannot_reach_pcie(tmp_path, monkeypatch):
    from benchmark.tpu_ops import run
    (tmp_path / "trial.json").write_text(json.dumps({"sdk": "unused"}))
    (tmp_path / "cmodel").mkdir()
    (tmp_path / "cmodel/result.json").write_text(json.dumps({"status": "numerical_mismatch"}))
    monkeypatch.setattr(run, "health", lambda *a:
                        (_ for _ in ()).throw(AssertionError("board accessed")))
    assert run.execute_trial(tmp_path, True) == "numerical_mismatch"


@pytest.mark.parametrize('admission_rejected', [False, True])
def test_new_board_program_cannot_enter_timing_after_canary_failure(tmp_path, monkeypatch, admission_rejected):
    from benchmark.tpu_ops import run, promote
    (tmp_path / 'trial.json').write_text(json.dumps(dict(sdk='unused', timeout=900)))
    (tmp_path / 'cmodel').mkdir()
    (tmp_path / 'cmodel/result.json').write_text(json.dumps(dict(status='passed')))

    def numerical(*args):
        if admission_rejected:
            raise NotImplementedError('known unsupported device lowering')
        return 'numerical_mismatch'

    monkeypatch.setattr(promote, 'canary', numerical)
    monkeypatch.setattr(run, 'supervised', lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError('repeated timing reached')))
    assert run.execute_trial(tmp_path, True) == (
        'unsupported' if admission_rejected else 'numerical_mismatch')
    assert not (tmp_path / 'pcie').exists()


def test_safety_exclusion_is_not_reported_as_missing_isa_or_lowering(tmp_path, monkeypatch):
    from benchmark.tpu_ops import run, promote
    from benchmark.tpu_ops.command_safety import BoardSafetyExclusion
    (tmp_path / 'trial.json').write_text(json.dumps(dict(sdk='unused')))
    (tmp_path / 'cmodel').mkdir()
    (tmp_path / 'cmodel/result.json').write_text(json.dumps(dict(status='passed')))

    def deny(*args):
        raise BoardSafetyExclusion('known unsafe long attention source')

    monkeypatch.setattr(promote, 'canary', deny)
    assert run.execute_trial(tmp_path, True) == 'blocked_device_safety'
    verdict = json.loads((tmp_path / 'pcie-canary/result.json').read_text())
    assert verdict['status'] == 'blocked_device_safety'
    assert not (tmp_path / 'pcie').exists()


@pytest.mark.parametrize('has_cmodel_proof', [False, True])
def test_explicit_safety_exclusion_stops_new_simulation_and_board_access(tmp_path, monkeypatch, has_cmodel_proof):
    from benchmark.tpu_ops import run
    trial = tmp_path / 'case/rv/tilelang/config'
    trial.mkdir(parents=True)
    (trial / 'trial.json').write_text(json.dumps(dict(sdk='unused')))
    (tmp_path / 'safety_exclusions.json').write_text(json.dumps({str(trial.resolve()):
        dict(skip_cmodel=True, reason='known unsafe family')}))
    if has_cmodel_proof:
        (trial / 'cmodel').mkdir()
        (trial / 'cmodel/result.json').write_text(json.dumps(dict(status='passed')))
    monkeypatch.setattr(run, 'supervised', lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError('native process reached')))
    assert run.execute_trial(trial, True) == (
        'blocked_device_safety' if has_cmodel_proof else 'skipped_device_safety')
    assert not (trial / 'pcie').exists()
    assert (trial / 'cmodel/result.json').exists() == has_cmodel_proof


def test_an_unrelated_pass_cannot_authorize_a_changed_trial(tmp_path, monkeypatch):
    from benchmark.tpu_ops import worker, build
    (tmp_path / "trial.json").write_text(json.dumps({"case": {"op": "matmul"}}))
    (tmp_path / "cmodel").mkdir()
    (tmp_path / "cmodel/result.json").write_text(
        json.dumps({
            "status": "passed",
            "trial_sha256": "different"
        }))
    monkeypatch.setattr(
        build, "build_runner", lambda *a:
        (_ for _ in ()).throw(AssertionError("compiled or accessed board")))
    monkeypatch.setattr(sys, "argv", ["worker", "--trial", str(tmp_path), "--runtime", "pcie"])
    assert worker.main() == 1
    result = json.loads((tmp_path / "pcie/result.json").read_text())
    assert "exact trial" in result["reason"]


def test_refined_memory_estimate_preserves_evidence_but_changed_contract_is_rejected(tmp_path, monkeypatch):
    from benchmark.tpu_ops import run, workloads, report
    case = {'id': 'fixture', 'op': 'elementwise-add', 'shape': [1, 32], 'dtype': 'float16'}
    estimate = [128]
    monkeypatch.setattr(workloads, 'cases', lambda: [case])
    monkeypatch.setattr(workloads, 'configs', lambda *args: [
        dict(block_rows=1, block_width=32, stages=1, estimated_local_bytes_per_lane=estimate[0])])
    monkeypatch.setattr(run, 'execute_trial', lambda *args: 'passed')
    monkeypatch.setattr(report, 'report', lambda *args: None)
    argv = ['run', '--output', str(tmp_path), '--sdk', str(tmp_path / 'sdk'),
            '--model', 'rv', '--implementation', 'tilelang']
    monkeypatch.setattr(sys, 'argv', argv)
    run.main()
    trial = next(tmp_path.glob('*/*/*/*/trial.json'))
    validated = trial.read_bytes()
    estimate[0] = 256
    run.main()
    assert trial.read_bytes() == validated
    monkeypatch.setattr(sys, 'argv', argv + ['--repeats', '11'])
    with pytest.raises(RuntimeError, match='specification changed'):
        run.main()
    assert trial.read_bytes() == validated


def test_refined_memory_estimate_does_not_reinterpret_a_model_ranked_tile(tmp_path, monkeypatch):
    from benchmark.tpu_ops import run, workloads
    case = dict(id='fixture', op='elementwise-add', shape=[1, 64], dtype='float16')
    old_space = [dict(block_rows=1, block_width=w, stages=1,
                      estimated_local_bytes_per_lane=128) for w in (32, 64)]
    new_space = [dict(item, estimated_local_bytes_per_lane=256) for item in old_space]
    recommendation = tmp_path / 'recommendations.json'
    recommendation.write_text(json.dumps({'fixture/rv/tilelang': [1]}))
    (tmp_path / 'spaces.json').write_text(json.dumps({'fixture/rv/tilelang': old_space}))
    monkeypatch.setattr(workloads, 'cases', lambda: [case])
    monkeypatch.setattr(workloads, 'configs', lambda *args: new_space)
    monkeypatch.setattr(sys, 'argv', ['run', '--output', str(tmp_path / 'run'), '--sdk', str(tmp_path),
                                    '--model', 'rv', '--implementation', 'tilelang', '--budget', '1',
                                    '--recommendations', str(recommendation), '--plan-only'])
    run.main()
    plan = json.loads((tmp_path / 'run/plan.json').read_text())
    assert len(plan) == 1
    assert plan[0]['spec']['config']['block_width'] == 64


def test_a_valid_cmodel_pass_cannot_bypass_quarantine(tmp_path, monkeypatch):
    from benchmark.tpu_ops import worker, build
    from tilelang.jit.adapter import tpu_profiling
    spec = tmp_path / 'trial.json'
    spec.write_text(json.dumps({'case': {'op': 'matmul', 'dtype': 'float16'}, 'seeds': [0]}))
    observed = {'passed': True, 'finite': True, 'atol': 0.0, 'rtol': 0.0, 'max_abs': 0.0}
    (tmp_path / 'cmodel').mkdir()
    (tmp_path / 'cmodel/result.json').write_text(
        json.dumps(
            dict(
                status='passed',
                trial_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),
                checks=[dict(observed, final=observed)])))
    monkeypatch.setattr(tpu_profiling, '_assert_pcie_device_not_quarantined', lambda *a:
                        (_ for _ in ()).throw(RuntimeError('quarantined-test')))
    monkeypatch.setattr(build, 'build_runner', lambda *a:
                        (_ for _ in ()).throw(AssertionError('board compiler reached')))
    monkeypatch.setattr(sys, 'argv', ['worker', '--trial', str(tmp_path), '--runtime', 'pcie'])
    assert worker.main() == 1
    assert 'quarantined-test' in json.loads((tmp_path / 'pcie/result.json').read_text())['reason']


def test_failed_final_measurement_is_not_hidden_by_search_pass(tmp_path):
    from benchmark.tpu_ops.report import report
    trial = tmp_path / "case" / "rv" / "tilelang" / "config"
    trial.mkdir(parents=True)
    (trial / "trial.json").write_text(
        json.dumps({
            "case": {
                "id": "case",
                "op": "matmul",
                "shape": [1, 32, 32],
                "dtype": "float16"
            },
            "model": "rv",
            "implementation": "tilelang",
            "config": {},
        }))
    for stage, result in (
        ("cmodel", {
            "status": "passed"
        }),
        ("pcie", {
            "status": "passed",
            "timing": {
                "p50": 1.0
            }
        }),
        ("pcie-final", {
            "status": "numerical_mismatch",
            "reason": "last repeated launch differs"
        }),
    ):
        (trial / stage).mkdir()
        (trial / stage / "result.json").write_text(json.dumps(result))
    report(tmp_path)
    row = json.loads((tmp_path / "all_trials.json").read_text())[0]
    assert row["timing_stage"] == "final"
    assert row["pcie_status"] == "numerical_mismatch"
    assert row["search_p50_us"] == 1.0
    assert json.loads((tmp_path / "best.json").read_text()) == []


def test_failed_board_canary_is_reported_without_a_performance_value(tmp_path):
    from benchmark.tpu_ops.report import report
    trial = tmp_path / 'case/rv/tilelang/config'
    trial.mkdir(parents=True)
    (trial / 'trial.json').write_text(json.dumps(dict(
        case=dict(id='case', op='matmul', shape=[1, 32, 32], dtype='float16'),
        model='rv', implementation='tilelang', config={})))
    for stage, status in (('cmodel', 'passed'), ('pcie-canary', 'numerical_mismatch')):
        (trial / stage).mkdir()
        (trial / stage / 'result.json').write_text(json.dumps(dict(status=status)))
    report(tmp_path)
    row = json.loads((tmp_path / 'all_trials.json').read_text())[0]
    assert row['cmodel_status'] == 'passed'
    assert row['pcie_status'] == 'numerical_mismatch'
    assert row['p50_us'] is None
    assert 'canary' in row['reason']
    assert json.loads((tmp_path / 'best.json').read_text()) == []


@pytest.mark.parametrize('completed_stage', [None, 'cmodel', 'pcie'])
def test_safety_exclusion_preserves_real_evidence_without_inventing_a_cmodel_run(tmp_path, completed_stage):
    from benchmark.tpu_ops.report import report
    trial = tmp_path / 'case/rv/tilelang/config'
    trial.mkdir(parents=True)
    (trial / 'trial.json').write_text(json.dumps(dict(
        case=dict(id='case', op='matmul', shape=[1, 32, 32], dtype='float16'),
        model='rv', implementation='tilelang', config={})))
    if completed_stage:
        (trial / 'cmodel').mkdir()
        (trial / 'cmodel/result.json').write_text(json.dumps(dict(status='passed')))
    if completed_stage == 'pcie':
        (trial / 'pcie').mkdir()
        (trial / 'pcie/result.json').write_text(json.dumps(dict(status='passed', timing=dict(p50=1.0))))
    (tmp_path / 'safety_exclusions.json').write_text(json.dumps({
        str(trial.resolve()): dict(skip_cmodel=True, reason='skip unsafe future launches')}))
    report(tmp_path)
    row = json.loads((tmp_path / 'all_trials.json').read_text())[0]
    assert row['cmodel_status'] == ('passed' if completed_stage else 'skipped_device_safety')
    assert (trial / 'cmodel/result.json').exists() == bool(completed_stage)
    assert row['safety_excluded'] == (completed_stage != 'pcie')
    assert row['p50_us'] == (1.0 if completed_stage == 'pcie' else None)


@pytest.mark.parametrize('status', ['numerical_mismatch', 'timeout', 'unsupported'])
def test_long_attention_promotion_review_rejects_failed_cmodel(tmp_path, status):
    from benchmark.tpu_ops.promote import inspect_trial
    (tmp_path / 'trial.json').write_text('{}')
    (tmp_path / 'cmodel').mkdir()
    (tmp_path / 'cmodel/result.json').write_text(json.dumps(dict(status=status)))
    with pytest.raises(ValueError, match='passing complete CModel'):
        inspect_trial(tmp_path)


def test_long_attention_promotion_review_rejects_unbounded_source(tmp_path):
    from benchmark.tpu_ops.promote import inspect_trial
    spec = dict(model='tpukernel', case=dict(op='flashattn', shape=[1, 8192, 32, 128]))
    path = tmp_path / 'trial.json'
    path.write_text(json.dumps(spec))
    (tmp_path / 'cmodel').mkdir()
    (tmp_path / 'cmodel/result.json').write_text(json.dumps(dict(
        status='passed', trial_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    (tmp_path / 'cmodel/kernel.c').write_text('void kernel(){}')
    with pytest.raises(NotImplementedError, match='bounded command epochs'):
        inspect_trial(tmp_path)


@pytest.mark.parametrize('implementation', ['tilelang', 'tilelang-matched', 'tilelang-optimized'])
def test_another_tile_cannot_bypass_the_long_rv_attention_fault_gate(implementation):
    from benchmark.tpu_ops.command_safety import require_board_source
    spec = dict(model='rv', implementation=implementation,
                case=dict(op='flashattn', shape=[1, 8192, 32, 128]))
    with pytest.raises(NotImplementedError, match='Do not replay another tile'):
        require_board_source(spec, 'void different_tile(){}', 'different source')


@pytest.mark.parametrize('op', ['rmsnorm', 'rmsnorm-splitk'])
def test_ppl_large_row_broadcast_is_denied_after_board_cmodel_disagreement(op):
    from benchmark.tpu_ops.command_safety import require_board_source
    spec = dict(model='rv', implementation='ppl', config=dict(block_rows=256),
                case=dict(op=op, shape=[2048, 4096]))
    with pytest.raises(NotImplementedError, match='only|failed FP16 and BF16'):
        require_board_source(spec, 'void kernel(){}', 'large-broadcast')
    spec['config']['block_rows'] = 64
    require_board_source(spec, 'void kernel(){}', 'original-lane-broadcast')


@pytest.mark.parametrize('known_hash', [
    'b856ceaa95a1a3d148ff843601567df3ca49630a20ea604cbdad309181b8661b',
    'd36fb2f8395416ffec523f7cb1a147ae6506d9c0ac1564acbb1bc93fa64b271b',
    '3bd6359a7325b32d531b3fe5b8fbdca2562e76f630c62b37020efd43169f1183',
])
def test_known_faulting_source_cannot_be_relabelled_as_a_small_problem(known_hash):
    from benchmark.tpu_ops.command_safety import require_board_source
    spec = dict(model='rv', implementation='tilelang',
                case=dict(op='flashattn', shape=[1, 128, 32, 128]))
    with pytest.raises(NotImplementedError, match='known faulting attention source'):
        require_board_source(spec, 'unchanged faulting code with different metadata', known_hash)


@pytest.mark.parametrize('implementation', ['ppl', 'tilelang', 'tilelang-matched', 'tilelang-optimized'])
def test_only_exact_previously_measured_long_tpukernel_programs_can_be_remeasured(implementation):
    from benchmark.tpu_ops.command_safety import (
        require_board_source, VERIFIED_LONG_TPUKERNEL_SOURCES, MARKER)
    sha = next(key for key, value in VERIFIED_LONG_TPUKERNEL_SOURCES.items()
               if value == implementation)
    spec = dict(model='tpukernel', implementation=implementation,
                config=dict(block_m=64, block_n=256, stages=1),
                case=dict(op='flashattn', shape=[1, 8192, 32, 128], dtype='float16', causal=False))
    audit = dict(source_sha256=sha, max_tiu_between_polls=20017, max_dma_between_polls=1552)
    require_board_source(spec, MARKER, sha, audit)
    # A passing proof at stages=1 cannot be relabelled as pipeline validation.
    spec['config']['stages'] = 2
    with pytest.raises(NotImplementedError, match='already-board-validated'):
        require_board_source(spec, MARKER, sha, audit)


@pytest.mark.parametrize('change', ['tile', 'dtype', 'causal', 'shape', 'source', 'implementation'])
def test_long_tpukernel_safety_cannot_be_inferred_from_another_program(change):
    from benchmark.tpu_ops.command_safety import (
        require_board_source, VERIFIED_LONG_TPUKERNEL_SOURCES, MARKER)
    sha = next(iter(VERIFIED_LONG_TPUKERNEL_SOURCES))
    spec = dict(model='tpukernel', implementation='ppl',
                config=dict(block_m=64, block_n=256, stages=1),
                case=dict(op='flashattn', shape=[1, 8192, 32, 128], dtype='float16', causal=False))
    if change == 'tile':
        spec['config']['block_m'] = 32
    if change == 'dtype':
        spec['case']['dtype'] = 'bfloat16'
    if change == 'causal':
        spec['case']['causal'] = True
    if change == 'shape':
        spec['case']['shape'][1] = 16384
    if change == 'source':
        sha = 'another complete-CModel-passing source'
    if change == 'implementation':
        spec['implementation'] = 'tilelang'
    audit = dict(source_sha256=sha, max_tiu_between_polls=20017, max_dma_between_polls=1552)
    with pytest.raises(NotImplementedError, match='already-board-validated'):
        require_board_source(spec, MARKER, sha, audit)


def test_faulted_postflight_is_quarantined_before_device_lock_release(tmp_path, monkeypatch):
    from benchmark.tpu_ops import run
    from tilelang.jit.adapter.tpu_profiling import TPUInstructionProfiler
    chip = {"status": "Fault", "mem_usage": "72MB"}
    recorded = []
    result_path = tmp_path / 'pcie/result.json'
    result_path.parent.mkdir()
    result_path.write_text(json.dumps({'status': 'passed', 'timing': {'p50': 1.0}}))
    monkeypatch.setattr(
        run.subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=json.dumps({"card0": {
                "chip0": chip
            }})))
    monkeypatch.setattr(TPUInstructionProfiler, "fail_closed_pcie_device",
                        lambda device, **kwargs: recorded.append((device, kwargs["reason"])))
    with pytest.raises(RuntimeError, match="device not idle/active"):
        run.health(tmp_path, "after", result_path=result_path)
    assert recorded and recorded[0][0] == 0
    assert "Fault" in recorded[0][1]
    assert json.loads((tmp_path / "health-after.json").read_text())["card0"]["chip0"] == chip
    assert json.loads(result_path.read_text())['status'] == 'device_fault'
    assert len(list(result_path.parent.glob('result-before-device-fault-*.json'))) == 1


def test_inflight_fault_stops_process_only_after_durable_quarantine(tmp_path, monkeypatch):
    from benchmark.tpu_ops import run
    from tilelang.jit.adapter.tpu_profiling import TPUInstructionProfiler
    events = []

    class Process:
        pid = 999999
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if self.returncode is None:
                raise run.subprocess.TimeoutExpired('fake-worker', timeout)
            return self.returncode

    process = Process()
    monkeypatch.setattr(run.subprocess, 'Popen', lambda *a, **k: process)
    monkeypatch.setattr(
        run.subprocess, 'run', lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({'card0': {
                'chip0': {
                    'status': 'Fault',
                    'mem_usage': '72MB'
                }
            }})))
    monkeypatch.setattr(TPUInstructionProfiler, 'fail_closed_pcie_device',
                        lambda *a, **k: events.append('quarantined'))

    def stop_group(pid, signal):
        assert pid == process.pid
        events.append('terminated')
        process.returncode = -15

    monkeypatch.setattr(run.os, 'killpg', stop_group)
    result = tmp_path / 'pcie/result.json'
    assert run.supervised(['fake-worker'], tmp_path / 'launch.log', {}, 900,
                          board_result=result) == 125
    assert events == ['quarantined', 'terminated']
    assert json.loads(result.read_text())['status'] == 'device_fault'


def test_large_legacy_source_is_denied_even_after_numerical_pass():
    from benchmark.tpu_ops.command_safety import require_board_source
    spec = {'model': 'tpukernel', 'case': {'op': 'rope', 'shape': [65536, 128]}}
    with pytest.raises(NotImplementedError, match='bounded command epochs'):
        require_board_source(spec, 'void kernel(){}', 'legacy')


def test_native_watchdog_exit_is_a_timeout_even_with_long_batch_budget(tmp_path):
    from benchmark.tpu_ops.build import command
    from subprocess import TimeoutExpired
    code = ('import sys; print("TPU_BENCH_ERROR=native launch/progress stalled for 120 seconds"); '
            'sys.exit(124)')
    with pytest.raises(TimeoutExpired) as observed:
        command([sys.executable, '-c', code], tmp_path, 'watchdog', timeout=3600)
    assert observed.value.timeout == 120
    assert 'stalled' in (tmp_path / 'watchdog.log').read_text()


def test_bounded_epoch_transform_is_outside_parallel_groups(tmp_path):
    from benchmark.tpu_ops.command_safety import bound_command_epochs, audit_command_epochs
    source = '''typedef struct { int unused; } API;
void inner(void) {
  for (int row=0; row<32; row++) {
    tpu_parallel_start();
    for (int k=0; k<4; k++) { tpu_bdc_dummy(); }
    tpu_parallel_end();
    tpu_gdma_cpy_L2S();
  }
}
int canary(const void *args) { API *api=(API*)args; inner(); tpu_poll(); return 0; }
'''
    case = {'op': 'flashattn', 'shape': [1, 2048, 32, 128]}
    transformed, _ = bound_command_epochs(source, case, 'tpukernel')
    assert transformed.index('++__bench_completed_tiles') > transformed.index('tpu_parallel_end')
    audit = audit_command_epochs(transformed, tmp_path, [])
    assert audit['tiu_api_emissions'] == 128
    assert audit['max_tiu_between_polls'] == 32
    assert audit['polls'] == 5


def test_bounded_epoch_transform_rejects_poll_inside_active_parallel_group():
    from benchmark.tpu_ops.command_safety import bound_command_epochs
    source = '''void inner(void) { tpu_parallel_start();
for(int r=0;r<32;r++) { tpu_gdma_cpy_L2S(); } tpu_parallel_end(); }
'''
    with pytest.raises(NotImplementedError, match='active parallel group'):
        bound_command_epochs(source, {'op': 'flashattn', 'shape': [1, 2048, 32, 128]}, 'tpukernel')


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("seed", [0, 17])
def test_long_attention_fixture_rejects_zero_output(tmp_path, dtype, seed):
    import torch
    from benchmark.tpu_ops.worker import fixtures, compare_output

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        # Heads are independent; one full 8192-token head checks the fixture's
        # signal magnitude without initializing an accelerator or vendor runtime.
        case = {"op": "flashattn", "shape": [1, 8192, 1, 128], "dtype": dtype}
        _, expected = fixtures(case, "tilelang", seed)
        path = tmp_path / "zero-output.bin"
        path.write_bytes(torch.zeros_like(expected).view(torch.uint8).numpy().tobytes())
        result = compare_output(path, expected, case)
        assert not result["passed"]
        assert result["mismatched"] > 0
    finally:
        torch.set_num_threads(threads)
