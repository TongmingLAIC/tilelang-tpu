"""Bound serial TPUKernel command epochs without changing tensor arithmetic.

This benchmark-local lowering workaround applies equally to the two frontends.
It is separate from algorithm optimization and must be validated in cmodel.
"""
import re
import json
import hashlib
import subprocess
from pathlib import Path

MARKER = "BENCH_BOUNDED_COMMAND_EPOCH_V1"
FAULT_SOURCE_SHA256 = "b856ceaa95a1a3d148ff843601567df3ca49630a20ea604cbdad309181b8661b"
RV_LONG_FAULT_SHA256 = "d36fb2f8395416ffec523f7cb1a147ae6506d9c0ac1564acbb1bc93fa64b271b"
TPUKERNEL_PIPELINE_FAULT_SHA256 = "3bd6359a7325b32d531b3fe5b8fbdca2562e76f630c62b37020efd43169f1183"
# These exact S8192 serial programs passed full CModel, board numerical
# canaries and 20 timing samples before the separate pipeline Fault.
# This permits their final remeasurement, not exploration of another tile.
VERIFIED_LONG_TPUKERNEL_SOURCES = {
    '1056c06bb88b65f3c76d1f601292b6ec1189ed9454c8824e47e91217a16885d1': 'ppl',
    '6acb9b17db234f8503e833505311855845b025a612ed2fd2dc43f18359e483cd': 'tilelang',
    '0c86061d70264c6ba7f84f7f544fc1ea3ea0cca032498ab2e2f920d556fa0691': 'tilelang-matched',
    '2de7808d76d1116bae4b27150f16e3d5f6f2a567b66436d39c1b2cdf62324892': 'tilelang-optimized',
}


class BoardSafetyExclusion(NotImplementedError):
    """A deliberate no-launch policy, distinct from unavailable lowering."""


def trial_exclusion(trial):
    trial = Path(trial).resolve()
    for root in trial.parents[:4]:
        path = root / 'safety_exclusions.json'
        if path.exists():
            return json.loads(path.read_text()).get(str(trial), {})
    return {}


def masked_c(source):
    pattern = r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(
        pattern, lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0]), source, flags=re.S)


def needs_bounded_epochs(case, model):
    if model != "tpukernel":
        return False
    if case.get('bounded_command_epochs'):
        return True
    op, shape = case['op'], case['shape']
    return ((op == 'flashattn' and shape[1] >= 2048) or (op == 'rope' and shape[0] > 1024) or
            (op in ('matmul', 'llama-mlp') and shape[0] >= 2048))


def bound_command_epochs(source, case, model):
    if not needs_bounded_epochs(case, model):
        return source, None
    if MARKER in source:
        raise ValueError("command epoch transform was already applied")
    code = masked_c(source)
    stack, spans = [], []
    for pos, char in enumerate(code):
        if char == '{':
            stack.append(pos)
        elif char == '}':
            spans.append((stack.pop(), pos))
    if stack:
        raise ValueError("unbalanced generated C")
    stores = list(re.finditer(r'\btpu_gdma_cpy_L2S\s*\(', code))
    if len(stores) != 1 and not (case['op'] == 'rope' and len(stores) == 2):
        raise NotImplementedError("large TPUKernel program requires one audited output-store site")
    ancestors = sorted((a, b) for a, b in spans if all(a < site.start() < b for site in stores))
    loops = [(a, b)
             for a, b in ancestors
             if re.search(r'\bfor\s*\([^{}]*\)\s*$', code[max(0, a - 220):a])]
    if not loops:
        raise NotImplementedError("cannot identify completed attention output-tile loop")
    _, end = loops[-1]
    functions = [(a, b)
                 for a, b in ancestors
                 if re.search(r'\b(?:void|int)\s+\w+\s*\([^{}]*\)\s*$', code[max(0, a - 1200):a])]
    if not functions:
        raise NotImplementedError("cannot locate attention device function")
    begin, _ = functions[-1]
    parallel = re.findall(r'\btpu_parallel_(start|end)\s*\(', code[begin:end])
    depth = 0
    for event in parallel:
        depth += 1 if event == 'start' else -1
        if depth < 0:
            raise NotImplementedError("unbalanced generated parallel groups")
    if depth:
        raise NotImplementedError("cannot poll inside an active parallel group")
    # Each query tile has at most 128 KV steps in this manifest. Eight completed
    # tiles leave a wide margin below the firmware's 0x40000 reset threshold.
    suffix = '\n      if ((++__bench_completed_tiles & 7u) == 0u) tpu_poll();\n'
    declaration = f'\n  /* {MARKER}: same policy for PPL and TileLang. */\n  unsigned __bench_completed_tiles = 0;\n'
    source = source[:end] + suffix + source[end:]
    source = source[:begin + 1] + declaration + source[begin + 1:]
    return source, dict(
        policy=MARKER,
        completed_output_tiles_per_poll=8,
        arithmetic_and_local_allocation_unchanged=True)


def require_board_source(spec, source, source_sha256, audit=None):
    if source_sha256 in (FAULT_SOURCE_SHA256, RV_LONG_FAULT_SHA256,
                         TPUKERNEL_PIPELINE_FAULT_SHA256):
        raise BoardSafetyExclusion("known faulting attention source is denied")
    case = spec["case"]
    if (spec['model'] == 'rv' and spec.get('implementation') == 'ppl' and
            case['op'] in ('rmsnorm', 'rmsnorm-splitk') and
            spec.get('config', {}).get('block_rows', 0) > 64):
        raise NotImplementedError(
            'SDK PPL RMSNorm uses one rvt_lane_bc for gamma: block_rows>64 '
            'failed FP16 and BF16 board canaries while full CModel passed. '
            'Skip this lowering; the original block_rows<=64 baseline remains valid.')
    if (spec['model'] == 'rv' and case['op'] == 'flashattn' and case['shape'][1] >= 8192 and
            spec.get('implementation') != 'ppl'):
        raise BoardSafetyExclusion(
            'S>=8192 TileLang/RV attention is disabled after a full-CModel-passing '
            'single-launch board Fault; compiler/runtime diagnosis is unresolved. '
            'Do not replay another tile or the optimized variant as a board probe.')
    if needs_bounded_epochs(case, spec['model']) and MARKER not in source:
        raise NotImplementedError(
            "large TPUKernel program requires bounded command epochs and fresh cmodel validation")
    if (needs_bounded_epochs(case, spec['model']) and
            (not audit or audit.get('source_sha256') != source_sha256 or max(
                audit.get('max_tiu_between_polls', 65536),
                audit.get('max_dma_between_polls', 65536)) >= 65536)):
        raise NotImplementedError('large TPUKernel program lacks a matching bounded-epoch audit')
    if (spec['model'] == 'tpukernel' and case['op'] == 'flashattn' and case['shape'][1] >= 8192 and
            not (
                source_sha256 in VERIFIED_LONG_TPUKERNEL_SOURCES and
                VERIFIED_LONG_TPUKERNEL_SOURCES[source_sha256] == spec.get('implementation') and
                case['shape'] == [1, 8192, 32, 128] and case.get('dtype') == 'float16' and
                not case.get('causal', False) and
                spec.get('config') == dict(block_m=64, block_n=256, stages=1))):
        raise BoardSafetyExclusion(
                'S>=8192 TPUKernel attention is restricted to the exact already-board-validated '
                'FP16 BM64/BN256 serial sources after the PPL pipeline Fault. '
                'A different tile, pipeline stage, dtype or algorithm needs independent evidence; '
                'do not probe it on this board.')


def audit_command_epochs(source, directory, include_flags):
    """Execute only scalar loop control on the host; remove all device calls.

    API emissions are a diagnostic, not an instruction-accurate simulator.
    The emitted trace links neither the device runtime nor the emulator.
    """
    directory = Path(directory)
    entry = re.search(r'int\s+(\w+)\s*\(const void \*\s*args\)', source)
    api = re.search(r'(\w+)\s*\*api\s*=\s*\(\w+\s*\*\)args', source)
    if not entry or not api:
        raise NotImplementedError('cannot audit generated device-entry ABI')
    trace = re.sub(r'^TPUKERNEL_FUNC_REGISTER\([^\n]*\)\s*$', '', source, flags=re.M)
    for name in sorted(set(re.findall(r'\b(tpu_\w+)\s*\(', masked_c(trace)))):
        pattern = re.compile(r'\b' + re.escape(name) + r'\s*\(')
        pos = 0
        while (match := pattern.search(masked_c(trace), pos)):
            depth, end = 1, match.end()
            while depth:
                depth += (trace[end] == '(') - (trace[end] == ')')
                end += 1
            value = '(scalar_t){.u32=0}' if name == 'tpu_cast' else '0'
            replacement = f'(bench_trace_record("{name}"), {value})'
            trace = trace[:match.start()] + replacement + trace[end:]
            pos = match.start() + len(replacement)
    prefix = r'''
#include <stdio.h>
#include <string.h>
static unsigned long long tiu=0,dma=0,mt=0,md=0,tt=0,td=0,polls=0;
static void bench_trace_record(const char *name) {
  if(strstr(name,"tpu_bdc_")){tiu++;tt++;if(tiu>mt)mt=tiu;}
  if(strstr(name,"tpu_gdma_")){dma++;td++;if(dma>md)md=dma;}
  if(!strcmp(name,"tpu_poll")){polls++;tiu=0;dma=0;}
}
'''
    suffix = f'int main(void){{{api[1]} a={{0}};{entry[1]}(&a);' + r'''
printf("{\"tiu_api_emissions\":%llu,\"dma_api_emissions\":%llu,\"polls\":%llu,\"max_tiu_between_polls\":%llu,\"max_dma_between_polls\":%llu}\n",tt,td,polls,mt,md);return 0;}
'''
    path = directory / 'command-epoch-trace.c'
    path.write_text(prefix + trace + suffix)
    program = directory / 'command-epoch-trace'
    build = subprocess.run(
        ['/usr/bin/cc', *include_flags, '-DUSING_CMODEL', '-O2',
         str(path), '-o',
         str(program)],
        capture_output=True,
        text=True,
        timeout=60)
    (directory / 'command-epoch-trace-build.log').write_text(build.stdout + build.stderr)
    if build.returncode:
        raise NotImplementedError('host-only command audit failed to compile; no board promotion')
    result = json.loads(subprocess.check_output([str(program)], text=True, timeout=30))
    result.update(
        method='host scalar loop trace; all device calls removed; no runtime linked',
        hardware_instruction_count=False,
        source_sha256=hashlib.sha256(source.encode()).hexdigest())
    (directory / 'command-epoch-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    if max(result['max_tiu_between_polls'], result['max_dma_between_polls']) >= 65536:
        raise NotImplementedError('command API burst exceeds conservative audit budget; skip board')
    return result
