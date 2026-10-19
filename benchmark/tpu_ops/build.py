"""Compile either frontend into the same native numerical/timing runner."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args, cwd, name, timeout=180):
    args = list(map(str, args))
    started = time.monotonic()
    with (Path(cwd) / "commands.jsonl").open("a") as log:
        log.write(
            json.dumps(dict(task=name, argv=args, cwd=str(cwd), timeout=timeout, event="start")) +
            "\n")
    try:
        with (Path(cwd) / (name + ".log")).open('w') as live_log:
            result = subprocess.run(
                args,
                cwd=cwd,
                text=True,
                stdout=live_log,
                stderr=subprocess.STDOUT,
                timeout=timeout)
    except subprocess.TimeoutExpired:
        with (Path(cwd) / "commands.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    dict(
                        task=name,
                        returncode=124,
                        wall_seconds=time.monotonic() - started,
                        event="timeout")) + "\n")
        raise
    with (Path(cwd) / "commands.jsonl").open("a") as log:
        log.write(
            json.dumps(
                dict(
                    task=name,
                    returncode=result.returncode,
                    wall_seconds=time.monotonic() - started,
                    event="finish")) + "\n")
    output = (Path(cwd) / (name + '.log')).read_text(errors='replace')
    if result.returncode == 124:
        limit = 120 if 'native launch/progress stalled' in output else timeout
        raise subprocess.TimeoutExpired(args, limit, output=output)
    if result.returncode:
        raise RuntimeError(f"{name} returned {result.returncode}: {output[-3500:]}")
    return output


def build_ppl(case, config, directory, sdk, model):
    from .ppl_sources import render
    directory = Path(directory)
    (directory / "kernel.pl").write_text(render(sdk, case, config))
    generated = directory / "generated"
    generated.mkdir()
    command([
        Path(sdk) / "bin/ppl-compile", directory / "kernel.pl", "--O3", "--chip=tpub_7_1_e",
        f"--o={generated}", *(["--rv"] if model == "rv" else [])
    ], directory, "ppl-compile")
    for folder, suffix in (("device", ".c"), ("host", ".cpp"), ("include", ".h")):
        src = generated / folder / ("kernel" + suffix)
        shutil.copy2(src, directory / src.name)
    return "bench", "bench_entry", 1


def build_tilelang(case, config, directory, sdk, model, runtime, variant):
    import tilelang
    import tvm
    from tilelang.jit.adapter.wrapper import TLTPUSourceWrapper
    from .tilelang_variants import build_program
    program = build_program(case, config, model, variant)
    target = tvm.target.Target(f"tpu -mcpu=bm1690e -tpu-programming-model={model}")
    artifact = tilelang.lower(
        program,
        target=target,
        runtime_mode=runtime,
        enable_host_codegen=False,
        enable_device_compile=False)
    TLTPUSourceWrapper(
        tvm.IRModule({"main": program}),
        artifact.kernel_source,
        target,
        output_indices=[len(program.params) - 1],
        output_dir=str(directory))
    source = (Path(directory) / "kernel.cpp").read_text()
    cores = int(re.search(r"int core_num = (\d+);", source).group(1))
    return "main_kernel", "main_kernel", cores


def build_runner(case, config, directory, sdk, model, runtime, implementation):
    from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    layout = resolve_ppl_layout(sdk, "bm1690e")
    if runtime == "pcie":
        # PPL's RV register allocator is nondeterministic across invocations.
        # Cross-compile the exact CModel-validated device program, rather than
        # asking either frontend to generate a second nominally equal program.
        previous = directory.parent / "cmodel"
        manifest = json.loads((previous / "build.json").read_text())
        for file in ("kernel.c", "kernel.cpp", "kernel.h"):
            shutil.copy2(previous / file, directory / file)
            if file in manifest.get("validated_sources", {}) and digest(
                    directory / file) != manifest["validated_sources"][file]:
                raise RuntimeError(f"validated cmodel source changed: {file}")
        if digest(directory / "kernel.c") != manifest["device_source_sha256"]:
            raise RuntimeError("validated cmodel source hash changed")
        name = "bench" if implementation == "ppl" else "main_kernel"
        entry = manifest["entry"]
        cores = manifest["cores"]
    elif implementation == "ppl":
        name, entry, cores = build_ppl(case, config, directory, sdk, model)
    else:
        name, entry, cores = build_tilelang(case, config, directory, sdk, model, runtime,
                                            implementation)
    # A compiler assertion stub is never submitted to the board.
    source = (directory / "kernel.c").read_text()
    command_epoch = None
    if runtime == "cmodel":
        from .command_safety import bound_command_epochs
        source, command_epoch = bound_command_epochs(source, case, model)
        (directory / "kernel.c").write_text(source)
    else:
        command_epoch = manifest.get("command_epoch")
    if re.search(r'TPUKERNEL_ASSERT\(\s*0\s*&&', source):
        raise NotImplementedError("vendor generated an unconditional TPUKERNEL_ASSERT stub")
    if case["op"] == "gptq-w4a16" and "tpu_bdc_f16_group_dequant" in source:
        symbols = subprocess.run(["nm", "-S", str(layout.firmware_archive)],
                                 text=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL,
                                 check=True).stdout
        evidence = "\n".join(
            line for line in symbols.splitlines() if "tpu_bdc_f16_group_dequant" in line)
        (directory / "dq2-firmware-symbols.txt").write_text(evidence + "\n")
        # This archive also contains a strong 108-byte definition. A weak
        # symbol alone would not establish which implementation is linked.
        objdump = layout.pcie_cross_gcc().with_name("riscv64-unknown-linux-gnu-objdump")
        disassembly = []
        members = subprocess.check_output(['ar', 't', str(layout.firmware_archive)], text=True).splitlines()
        # This archive has duplicate tpu_kernel.c.o member names. ar p is
        # ambiguous; inspect each occurrence independently, never concatenated ELFs.
        for member in ("tpu_kernel_common.c.o", "tpu_kernel.c.o"):
            for occurrence in range(1, members.count(member) + 1):
                with tempfile.TemporaryDirectory(dir=directory) as temporary:
                    subprocess.run(['ar', 'xN', str(occurrence), str(layout.firmware_archive), member],
                                   cwd=temporary, check=True, capture_output=True)
                    obj = directory / f'dq2-{occurrence}-{member}'
                    shutil.copy2(Path(temporary) / member, obj)
                text = subprocess.check_output(
                    [str(objdump), '-dr', '--disassemble=tpu_bdc_f16_group_dequant', str(obj)], text=True)
                if '<tpu_bdc_f16_group_dequant>:' in text:
                    disassembly.append(text)
        (directory / "dq2-firmware-disassembly.txt").write_text("\n".join(disassembly))
        if disassembly and all("tp_debug" in text for text in disassembly):
            raise NotImplementedError(
                "TPU-Kernel DQ2 firmware weak/strong definitions are diagnostic stubs (110/108 bytes, debug calls; disassembly retained); no board promotion. "
                + evidence)
    header = (directory / "kernel.h").read_text()
    (directory / "chip_map.h").write_text(
        '#pragma once\n#include <unordered_map>\n#include <string>\n'
        'static std::unordered_map<std::string,std::string> CHIP_MAP={{"bm1690e","tpub_7_1_e"}};\n')
    match = re.search(rf"fill_{name}_struct\((.*?)\);", header, re.S)
    params = match.group(1).split(",") if match else []
    if not params or any("unsigned long long" not in p for p in params):
        raise RuntimeError("fixed-shape benchmark ABI must contain pointers only")
    fields = [p.strip().split()[-1] for p in params]
    fill_args = ",".join(f"(unsigned long long)ptrs[{i}]" for i in range(len(fields)))
    binding = f'''#include "kernel.cpp"
static std::vector<char> prepare_api(const std::vector<void *> &ptrs) {{
  if(ptrs.size() != {len(fields)}) throw std::runtime_error("pointer count");
  auto api=fill_{name}_struct({fill_args});
  int status={name}_check_mem_s(&api);
  if(status) throw std::runtime_error("memory planner: " + std::to_string(status));
  std::vector<char> bytes(sizeof(api)*{cores});
  for(int i=0;i<{cores};++i) std::memcpy(bytes.data()+i*sizeof(api),&api,sizeof(api));
  return bytes;
}}
static const char *entry_name() {{ return "{entry}"; }}
static int group_count() {{ return 1; }}
static int core_count() {{ return {cores}; }}
'''
    (directory / "binding.inc").write_text(binding)
    definitions = [f"-D{x}" for x in layout.compile_definitions]
    if model == "rv":
        definitions += ["-DTILELANG_TPU_RV"]
    else:
        definitions += ["-DTILELANG_TPU_TPUKERNEL"]
    includes = [f"-I{x}" for x in layout.include_dirs_for(runtime)]
    includes += [f"-I{directory}", f"-I{layout.tpudnn_include}"]
    shared_flags = [*definitions, *includes, "-O3", "-fPIC"]
    if runtime == 'cmodel' and command_epoch:
        from .command_safety import audit_command_epochs
        command_epoch['audit'] = audit_command_epochs(source, directory, definitions + includes)
    if runtime == "cmodel":
        cc = "/usr/bin/cc"
        flags = [*shared_flags, "-DUSING_CMODEL"]
        rpath = f"{layout.runtime_lib}:{layout.backend_lib}"
    else:
        cc = str(layout.pcie_cross_gcc())
        flags = [*shared_flags, "-flto"]
        rpath = f"{layout.pcie_runtime_lib()}:{layout.backend_lib}"
    for src, obj in ((directory / "kernel.c", "device.o"), (layout.ppl_helper_source, "helper.o")):
        command([cc, *flags, "-c", src, "-o", directory / obj], directory, obj + "-compile")
    link = [
        cc, "-shared", "-fPIC", "-Wl,--no-undefined", "-o", directory / "module.so",
        directory / "device.o", directory / "helper.o"
    ]
    if runtime == "cmodel":
        link += [str(layout.emulator_library), f"-Wl,-rpath,{rpath}"]
    else:
        link += [
            "-flto", "-Wl,--whole-archive",
            str(layout.firmware_archive), "-Wl,--no-whole-archive", "-ldl"
        ]
    command(link + ["-lm"], directory, "device-link")
    # Same host code, allocation and launch ABI for both frontends.
    host_flags = [*shared_flags, "-std=c++17"]
    if runtime == "cmodel":
        host_flags += ["-DUSING_CMODEL"]
    runtime_lib = layout.runtime_lib if runtime == "cmodel" else layout.pcie_runtime_lib()
    command([
        "/usr/bin/c++", *host_flags, HERE / "native_runner.cpp", "-include", "cstring", "-o",
        directory / "runner", f"-L{runtime_lib}", f"-L{layout.backend_lib}",
        f"-Wl,--disable-new-dtags,-rpath,{rpath}", "-ltpuv7_rt", "-ltpudnn", "-lpthread", "-ldl"
    ], directory, "host-compile")
    canonical = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    canonical = re.sub(r"\s+", "", canonical)
    result = {
        "device_source_sha256": digest(directory / "kernel.c"),
        "canonical_device_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "runner_source_sha256": digest(HERE / "native_runner.cpp"),
        "sdk": str(Path(sdk).resolve()),
        "model": model,
        "runtime": runtime,
        "implementation": implementation,
        "entry": entry,
        "cores": cores,
        "module_sha256": digest(directory / "module.so")
    }
    result["command_epoch"] = command_epoch
    from .ppl_sources import SOURCES
    relative = SOURCES.get(case["op"], "examples/cxx/arith/add_broadcast.pl")
    original = Path(sdk) / relative
    if original.exists():
        result["baseline_source"] = {"path": str(original), "sha256": digest(original)}
    result["pipeline"] = {
        "requested":
            config.get("stages", 1),
        "interpretation":
            "compiler-managed" if implementation == "ppl" and config.get("stages") == 2 else
            "manual double buffer" if config.get("stages") == 2 else "serial",
        "parallel_start_occurrences":
            len(re.findall(r"(?:tpu_parallel_start|rvt_parallel_start)\s*\(", source))
    }
    result["benchmark_sources"] = {path.name: digest(path) for path in HERE.glob("*.py")}
    algorithm_root = HERE.parents[1] / 'tpu_demo'
    result['algorithm_sources'] = {
        str(path.relative_to(algorithm_root)): digest(path)
        for path in algorithm_root.rglob('*.py')
    }
    if implementation == 'tilelang-optimized':
        optimized_root = HERE.parents[1] / 'tpu_benchmark'
        result['optimized_sources'] = {
            str(path.relative_to(optimized_root)): digest(path)
            for path in optimized_root.rglob('*.py')
        }
    result["validated_sources"] = {
        name: digest(directory / name) for name in ("kernel.c", "kernel.cpp", "kernel.h")
    }
    result["ppl_compiler_sha256"] = digest(Path(sdk) / "bin/ppl-compile")
    result["ppl_math_helpers_sha256"] = digest(Path(sdk) / "inc/ppl_wrapper_func.h")
    if case['op'] == 'rope' and config.get('expression') == 'pair_channels':
        pair_count = case['shape'][0] * case['shape'][1] // 2
        inner = min(pair_count, 32768)
        result['pair_channel_layout'] = {
            'logical_problem': case['shape'], 'global_view': [pair_count // inner, inner, 1, 2],
            'compact_coefficient_view': [pair_count // inner, inner, 1, 1],
            'local_shape': [config['block_rows'] * case['shape'][1] // 2, 2],
            'flat_bytes_unchanged': True, 'arithmetic': 'original FP32 pair formula',
        }
    elif case["op"] == "rope" and case["shape"][0] > 65535 and implementation != "ppl":
        result["global_axis_adapter"] = {
            "logical_problem": case["shape"],
            "view": [case["shape"][0] // 32768, 32768, 1, case["shape"][1]],
            "flat_bytes_and_local_tiles_unchanged": True,
        }
    if case["op"] == "paged-attention":
        result["baseline_components"] = {
            name: digest(Path(sdk) / name)
            for name in ("examples/cxx/llm/flash_attention_GQA_multicore.pl",
                         "examples/cxx/llm/tgi/embedding.pl")
        }
    (directory / "build.json").write_text(json.dumps(result, indent=2))
    return result
