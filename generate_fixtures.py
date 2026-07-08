import json
from pathlib import Path

import numpy as np

from packing import elements_per_word, pack
from quantize import quantize
from reference import reference_matmul

"""
For given sizes, write weights/scales/activations + correctness reference to disk
"""


DTYPES = {
    "activations": "float16",
    "packed_qweight": "uint32",
    "scales": "float32",
    "output": "float16",
}


def fixture_name(
    M: int, N: int, K: int, group_size: int, num_bits: int, seed: int
) -> str:
    return f"M{M}_N{N}_K{K}_g{group_size}_b{num_bits}_s{seed}"


def generate_fixture(
    M: int, N: int, K: int, group_size: int, seed: int, num_bits: int = 4
) -> dict:
    """
    Build one fixture bundle from scratch.

    Raises ValueError early (with a clear message) if the shape parameters
    are inconsistent, rather than letting quantize()/pack() fail cryptically
    deeper in the pipeline.
    """
    epw = elements_per_word(num_bits)
    if K % group_size != 0:
        raise ValueError(f"K={K} not divisible by group_size={group_size}")
    if K % epw != 0:
        raise ValueError(
            f"K={K} not divisible by elements_per_word={epw} (num_bits={num_bits})"
        )

    rng = np.random.default_rng(seed)
    activations = rng.normal(size=(M, K)).astype(np.float32)
    w_true = rng.normal(size=(K, N)).astype(np.float32)

    qweight, scales = quantize(w_true, group_size, num_bits=num_bits)
    packed_qweight = pack(qweight, num_bits=num_bits)
    output = reference_matmul(
        activations, packed_qweight, scales, group_size, num_bits=num_bits
    )

    meta = dict(
        M=M,
        N=N,
        K=K,
        group_size=group_size,
        num_bits=num_bits,
        seed=seed,
        num_words=packed_qweight.shape[0],
        num_groups=scales.shape[0],
        dtypes=DTYPES,
    )

    return dict(
        activations=activations.astype(np.float16),
        packed_qweight=packed_qweight,
        scales=scales.astype(np.float32),
        output=output,
        meta=meta,
    )


def save_fixture(fixture: dict, out_dir: Path) -> None:
    """Write a fixture's arrays as raw binary + meta.json into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in ("activations", "packed_qweight", "scales", "output"):
        arr = np.ascontiguousarray(fixture[key])
        arr.tofile(out_dir / f"{key}.bin")

    with open(out_dir / "meta.json", "w") as f:
        json.dump(fixture["meta"], f, indent=2)


def load_fixture(fixture_dir: Path) -> dict:
    """Read back a fixture written by save_fixture(). Python-side loader
    (e.g. for pytest); the CUDA benchmark reads meta.json + the .bin files
    directly rather than going through this function."""
    fixture_dir = Path(fixture_dir)
    with open(fixture_dir / "meta.json") as f:
        meta = json.load(f)

    M, N, K = meta["M"], meta["N"], meta["K"]
    num_words, num_groups = meta["num_words"], meta["num_groups"]

    activations = np.fromfile(
        fixture_dir / "activations.bin", dtype=np.float16
    ).reshape(M, K)
    packed_qweight = np.fromfile(
        fixture_dir / "packed_qweight.bin", dtype=np.uint32
    ).reshape(num_words, N)
    scales = np.fromfile(fixture_dir / "scales.bin", dtype=np.float32).reshape(
        num_groups, N
    )
    output = np.fromfile(fixture_dir / "output.bin", dtype=np.float16).reshape(M, N)

    return dict(
        activations=activations,
        packed_qweight=packed_qweight,
        scales=scales,
        output=output,
        meta=meta,
    )


DEFAULT_FIXTURES = [
    dict(M=16, N=16, K=128, group_size=128, num_bits=4, seed=0),
    dict(M=64, N=128, K=512, group_size=128, num_bits=4, seed=0),
    dict(M=128, N=256, K=1024, group_size=128, num_bits=4, seed=1),
]

# Boundary-case correctness fixtures. Small and cheap on purpose -- these exist to exercise
# tile-edge / partial-tile code paths (padding, masking, predicated writes), not to be
# representative of real workload scale. Add these once Stage 3a's TM/TN thread tiling is in,
# since that's the code most likely to introduce an off-by-one at a boundary that a
# clean-multiple-of-everything fixture would never touch.
#   - M17: M is not a multiple of typical BM/TM tile sizes (17 is prime; won't divide evenly
#     into any power-of-two tile).
#   - N100: same idea for N/TN, using a non-power-of-two, non-multiple-of-32 value.
#   - K192_g64: K spans 3 groups at group_size=64 (K % group_size == 0 but K > group_size,
#     i.e. multiple chunks-per-group within one fixture, distinct from the K==group_size case
#     your existing fixtures happen to hit).
BOUNDARY_FIXTURES = [
    dict(M=17, N=128, K=256, group_size=128, num_bits=4, seed=2),
    dict(M=64, N=100, K=256, group_size=128, num_bits=4, seed=2),
    dict(M=64, N=128, K=192, group_size=64, num_bits=4, seed=2),
]

# Benchmark-scale fixtures: realistic LLM prefill/decode sizes, plus the shapes already used for
# the Marlin reference sweep (reference/marlin/marlin_speed_bench.cu) so the cumulative
# GFLOPS-vs-stage chart is comparing apples to apples. These DO go through the full oracle --
# at these sizes generate_fixture() costs a few seconds each (the dequantized fp32 weight
# matrix for the 8192x8192 case is ~256MB, well within reach), so there's no reason to skip
# real verification here the way we did for the Marlin-only speed harness.
#
# NOTE: two group_size=-1 (ungrouped / per-column scale) shapes from the original sweep are
# omitted below -- generate_fixture() requires K % group_size == 0, which -1 breaks. Add them
# back once/if the quantize/pack pipeline has a defined ungrouped mode.
#
# Hidden dims (4096, 11008, 8192) match Llama-family intermediate/hidden sizes. M sweep covers
# decode (M=1, single-token autoregressive step) through prefill (M=1024, large-batch/long-
# context prompt processing).
BENCHMARK_SHAPES = [
    # Decode-regime: small M, memory-bandwidth-bound rather than compute-bound.
    dict(M=1, N=4096, K=4096, group_size=128, num_bits=4, seed=0),
    dict(M=16, N=4096, K=4096, group_size=128, num_bits=4, seed=0),
    # Mid-batch: transition zone between memory- and compute-bound.
    dict(M=64, N=4096, K=4096, group_size=128, num_bits=4, seed=0),
    dict(M=256, N=4096, K=4096, group_size=128, num_bits=4, seed=0),
    # Prefill-regime: large M, compute-bound.
    dict(M=1024, N=4096, K=4096, group_size=128, num_bits=4, seed=0),
    # Wider FFN-style rectangular shapes (e.g. gate/up projections).
    dict(M=16, N=4096, K=11008, group_size=128, num_bits=4, seed=0),
    dict(M=256, N=4096, K=11008, group_size=128, num_bits=4, seed=0),
    # Larger square-ish shapes.
    dict(M=16, N=8192, K=8192, group_size=128, num_bits=4, seed=0),
    dict(M=256, N=8192, K=8192, group_size=128, num_bits=4, seed=0),
]


def generate_all(out_root: Path = Path("fixtures"), specs=None) -> list:
    """Generate and save every fixture in `specs` (default DEFAULT_FIXTURES).
    Returns the list of output directories written."""
    specs = specs if specs is not None else DEFAULT_FIXTURES
    out_root = Path(out_root)
    written = []
    for spec in specs:
        name = fixture_name(
            spec["M"],
            spec["N"],
            spec["K"],
            spec["group_size"],
            spec["num_bits"],
            spec["seed"],
        )
        fixture = generate_fixture(**spec)
        out_dir = out_root / name
        save_fixture(fixture, out_dir)
        written.append(out_dir)
    return written


if __name__ == "__main__":
    import argparse
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fixture = generate_fixture(
            M=16, N=16, K=128, group_size=128, seed=0, num_bits=4
        )
        out_dir = Path(tmp) / fixture_name(16, 16, 128, 128, 4, 0)
        save_fixture(fixture, out_dir)
        loaded = load_fixture(out_dir)

        for key in ("activations", "packed_qweight", "scales", "output"):
            assert np.array_equal(fixture[key], loaded[key]), (
                f"save/load round-trip mismatch on '{key}'"
            )

        recomputed = reference_matmul(
            loaded["activations"],
            loaded["packed_qweight"],
            loaded["scales"],
            loaded["meta"]["group_size"],
            num_bits=loaded["meta"]["num_bits"],
        )
        assert np.array_equal(recomputed, loaded["output"]), (
            "loaded output does not match reference_matmul(loaded inputs)"
        )

        print("fixture save/load round-trip + consistency check passed.")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("fixtures"))
    parser.add_argument(
        "--clean", action="store_true", help="remove --out-dir before generating"
    )
    parser.add_argument(
        "--include-boundary",
        action="store_true",
        help="also generate BOUNDARY_FIXTURES (tile-edge / partial-tile cases)",
    )
    parser.add_argument(
        "--include-benchmark",
        action="store_true",
        help="also generate BENCHMARK_SHAPES (prefill/decode-scale, oracle-verified; slower, "
        "biggest fixture ~256MB dequantized weight matrix in-memory during generation)",
    )
    args = parser.parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    specs = list(DEFAULT_FIXTURES)
    if args.include_boundary:
        specs += BOUNDARY_FIXTURES
    if args.include_benchmark:
        specs += BENCHMARK_SHAPES

    written = generate_all(args.out_dir, specs=specs)
    print(f"wrote {len(written)} fixtures to {args.out_dir}/:")
    for d in written:
        print(f"  {d}")
