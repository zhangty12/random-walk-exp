from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence


def load_extension(
    name: str = "binary_search_random_walk_ext",
    verbose: bool = False,
    extra_cuda_cflags: Optional[Sequence[str]] = None,
) -> Any:
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        raise SystemExit("PyTorch is required to build the custom CUDA extension.") from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to build and run the custom random-walk extension.")

    root = Path(__file__).resolve().parent
    sources = [
        str(root / "csrc" / "binary_search_random_walk.cpp"),
        str(root / "csrc" / "binary_search_random_walk_cuda.cu"),
    ]
    cuda_flags = ["-O3", "--use_fast_math"]
    if extra_cuda_cflags:
        cuda_flags.extend(extra_cuda_cflags)

    return load(
        name=name,
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=cuda_flags,
        with_cuda=True,
        verbose=verbose,
    )
