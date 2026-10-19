# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""TPU target capabilities and compilation configuration.

The public selection is deliberately split into three independent axes:

* chip: the physical TPU generation (``bm1690`` or ``bm1690e``);
* programming model: the device ISA/API family (``tpukernel`` or ``rv``);
* runtime mode: the host execution environment (``cmodel`` or ``pcie``).

The TVM Target is the single compilation identity. A complete target uses
``tpu -mcpu=<chip> -tpu-programming-model=<model>``. Runtime selection remains
outside the Target because it changes how an already selected device program
is hosted, not which program is generated.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional, Tuple

TPUProgrammingModel = Literal["tpukernel", "rv"]
TPURuntimeMode = Literal["pcie", "cmodel"]


@dataclass(frozen=True)
class TPUChipSpec:
    """Stable capability description for one supported physical TPU chip."""

    name: str
    ppl_arch: str
    ppl_compile_definitions: Tuple[str, ...]
    physical_core_count: int
    programming_models: Tuple[TPUProgrammingModel, ...]

    def supports(self, programming_model: str) -> bool:
        return programming_model in self.programming_models


# BM1690 and BM1690E share the TPUv7 local-memory geometry used by the current
# TileLang allocator/codegen. The different SDK architecture and physical core
# count are represented here rather than inferred from a directory layout.
TPU_CHIP_SPECS: Mapping[str, TPUChipSpec] = MappingProxyType({
    "bm1690":
        TPUChipSpec(
            name="bm1690",
            ppl_arch="tpub_7_1",
            ppl_compile_definitions=("__tpub_7_1__", "__sg2260__"),
            physical_core_count=8,
            programming_models=("tpukernel",),
        ),
    "bm1690e":
        TPUChipSpec(
            name="bm1690e",
            ppl_arch="tpub_7_1_e",
            ppl_compile_definitions=("__tpub_7_1_e__", "__bm1690e__"),
            physical_core_count=4,
            programming_models=("tpukernel", "rv"),
        ),
})


def _normalize_chip_name(chip: str) -> str:
    if not isinstance(chip, str):
        raise TypeError(f"TPU chip must be a string, got {type(chip).__name__}")
    normalized = chip.strip().lower()
    if not normalized:
        raise ValueError("TPU chip must not be empty")
    return normalized


def get_tpu_chip_spec(chip: str) -> TPUChipSpec:
    """Return a supported chip capability record or fail before toolchain use."""
    normalized = _normalize_chip_name(chip)
    try:
        return TPU_CHIP_SPECS[normalized]
    except KeyError as exc:
        supported = ", ".join(TPU_CHIP_SPECS)
        raise ValueError(f"Unsupported TPU chip {chip!r}; supported chips: {supported}") from exc


def _validate_programming_model(programming_model: str) -> TPUProgrammingModel:
    if programming_model not in ("tpukernel", "rv"):
        raise ValueError("Unsupported TPU programming model "
                         f"{programming_model!r}; expected 'tpukernel' or 'rv'")
    return programming_model


def get_tpu_target_chip(target: Any) -> Optional[str]:
    """Read the chip selected by a TPU ``Target``.

    ``-mcpu`` is the only physical-chip selector. ``Target.model`` remains
    ordinary workload metadata and is never interpreted as hardware.
    """
    kind = getattr(getattr(target, "kind", None), "name", None)
    if kind != "tpu":
        return None

    attrs = getattr(target, "attrs", {})
    raw_mcpu = attrs.get("mcpu")
    mcpu_value = str(raw_mcpu).strip() if raw_mcpu is not None else ""
    if not mcpu_value or mcpu_value == "unknown":
        raise ValueError("TPU target requires an explicit physical chip: "
                         "tpu -mcpu=<bm1690|bm1690e> "
                         "-tpu-programming-model=<tpukernel|rv>")
    return get_tpu_chip_spec(mcpu_value).name


def get_tpu_target_programming_model(target: Any) -> Optional[TPUProgrammingModel]:
    """Read the required programming model from a TPU Target."""
    kind = getattr(getattr(target, "kind", None), "name", None)
    if kind != "tpu":
        return None
    attrs = getattr(target, "attrs", {})
    raw_model = attrs.get("tpu-programming-model")
    model = str(raw_model).strip() if raw_model is not None else ""
    if not model or model == "unknown":
        raise ValueError("TPU target requires an explicit programming model: "
                         "-tpu-programming-model=<tpukernel|rv>")
    return _validate_programming_model(model)


@dataclass(frozen=True)
class TPUTargetSpec:
    """Compile-time TPU identity carried by a complete TVM Target."""

    chip: str
    programming_model: TPUProgrammingModel

    def __post_init__(self):
        spec = get_tpu_chip_spec(self.chip)
        programming_model = _validate_programming_model(self.programming_model)
        if not spec.supports(programming_model):
            supported = ", ".join(spec.programming_models)
            raise ValueError(f"TPU chip {spec.name!r} does not support programming model "
                             f"{programming_model!r}; supported models: {supported}")
        object.__setattr__(self, "chip", spec.name)
        object.__setattr__(self, "programming_model", programming_model)

    @property
    def chip_spec(self) -> TPUChipSpec:
        return get_tpu_chip_spec(self.chip)


@dataclass(frozen=True)
class TPURuntimeConfig:
    """Host-side execution selection, intentionally absent from codegen identity."""

    runtime_mode: TPURuntimeMode = "cmodel"

    def __post_init__(self):
        if self.runtime_mode not in ("pcie", "cmodel"):
            raise ValueError(f"Unsupported TPU runtime mode {self.runtime_mode!r}; "
                             "expected 'pcie' or 'cmodel'")


def resolve_tpu_target(*, target: Any) -> TPUTargetSpec:
    """Resolve the compile-time identity from one complete TPU Target."""
    selected_chip = get_tpu_target_chip(target)
    programming_model = get_tpu_target_programming_model(target)
    if selected_chip is None or programming_model is None:
        raise ValueError("resolve_tpu_target requires a TPU Target")
    return TPUTargetSpec(
        chip=selected_chip,
        programming_model=programming_model,
    )


def resolve_tpu_runtime(*, runtime_mode: Optional[TPURuntimeMode] = None) -> TPURuntimeConfig:
    """Resolve host execution without changing the compiled device program.

    CModel is the safe default: omitting a runtime choice must never
    initialize a physical board. PCIe loading has an additional explicit gate
    in the adapter.
    """
    if runtime_mode is None:
        runtime_mode = "cmodel"
    return TPURuntimeConfig(runtime_mode=runtime_mode)
