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
    args = parser.parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    written = generate_all(args.out_dir)
    print(f"wrote {len(written)} fixtures to {args.out_dir}/:")
    for d in written:
        print(f"  {d}")
