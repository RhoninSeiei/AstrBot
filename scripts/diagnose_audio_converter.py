"""Print bounded, non-sensitive diagnostics for the native audio converter."""

import asyncio
import importlib.util
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


def report(label: str, value: object) -> None:
    print(json.dumps({label: value}, sort_keys=True), flush=True)


def process_status() -> dict[str, str]:
    status = Path("/proc/self/status")
    if not status.is_file():
        return {}
    return {
        key: value.strip()
        for line in status.read_text().splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
        if key in {"Threads", "VmSize", "VmRSS"}
    }


def main() -> None:
    report(
        "parent",
        {
            "limits": {
                name: resource.getrlimit(getattr(resource, name))
                for name in ("RLIMIT_STACK", "RLIMIT_AS", "RLIMIT_NPROC")
            },
            "cpu_count": os.cpu_count(),
            "affinity_count": len(os.sched_getaffinity(0)),
            "process_status": process_status(),
            "numeric_thread_environment": {
                name: value
                for name in (
                    "OMP_NUM_THREADS",
                    "OMP_THREAD_LIMIT",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "MALLOC_ARENA_MAX",
                )
                if (value := os.environ.get(name, "")).isdigit()
            },
        },
    )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required for converter diagnostics")
    version = subprocess.run(
        [ffmpeg, "-version"], capture_output=True, text=True, check=True, timeout=5
    )
    report("ffmpeg", version.stdout.splitlines()[0])
    libraries = subprocess.run(
        ["ldd", ffmpeg], capture_output=True, text=True, check=True, timeout=5
    )
    report(
        "thread_libraries",
        [
            line.strip()
            for line in libraries.stdout.splitlines()
            if any(
                name in line.lower() for name in ("libgomp", "libomp", "blas", "mkl")
            )
        ],
    )

    module_path = (
        Path(__file__).resolve().parents[1]
        / "astrbot/core/provider/oauth/openai_oauth_audio_input.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audio_converter_diagnostics", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    budget = 512 * 1024 * 1024
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            module._LIMITED_EXEC_SCRIPT,
            str(512 * 1024 + 64 * 1024),
            str(budget),
            sys.executable,
            "-c",
            "import json,resource; print(json.dumps({name: "
            "resource.getrlimit(getattr(resource,name)) for name in "
            "('RLIMIT_STACK','RLIMIT_AS','RLIMIT_FSIZE')}))",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    report(
        "limited_child",
        {
            "returncode": child.returncode,
            "stdout": child.stdout[:1024],
            "stderr": child.stderr[:1024],
        },
    )
    with tempfile.TemporaryDirectory(prefix="astrbot_audio_diagnostics_") as directory:
        root = Path(directory)
        source_wav = root / "source.wav"
        source_ogg = root / "source.ogg"
        with wave.open(str(source_wav), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(bytes(3200))
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(source_wav), str(source_ogg)],
            capture_output=True,
            check=True,
            timeout=5,
        )
        resolver = module.BoundedOAuthAudioResolver(
            str(source_ogg), max_bytes=512 * 1024, timeout=5
        )
        asyncio.run(diagnose_conversion(resolver))


async def diagnose_conversion(resolver) -> None:
    try:
        async with resolver.as_wav_path() as resolved:
            with wave.open(str(resolved.path), "rb") as output:
                report(
                    "conversion",
                    {
                        "result": "success",
                        "channels": output.getnchannels(),
                        "sample_rate": output.getframerate(),
                        "sample_width": output.getsampwidth(),
                        "frames": output.getnframes(),
                    },
                )
    except ValueError as error:
        # Keep the normal test suite responsible for gating conversion failures.
        report("conversion", {"result": "error", "cause": str(error.__cause__)[:1024]})


if __name__ == "__main__":
    main()
