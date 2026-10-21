"""Run existing tests using the parameters in a TOML file."""

import argparse
import os
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("all", "check", "bench", "tail", "long", "capacity")


def load_config(path, root=ROOT, stage=None):
    config = tomllib.loads(path.read_text())
    for key in ("model", "output_dir"):
        config[key] = str((root / Path(os.path.expandvars(config[key])).expanduser()).resolve())
    if stage:
        config["stage"] = stage
    return config


def run_logged(name, command, output, env):
    path = output / f"{name}.log"
    print(f"{name}: {path}", flush=True)
    with (
        path.open("w") as log,
        subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process,
    ):
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def execute(config, output):
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": config["gpu"],
        "NANOVLLM_MODEL": config["model"],
        "NANOVLLM_DTYPE": config["dtype"],
        "REPEATS": str(config["repeats"]),
        "PYTHON": sys.executable,
        "PYTHONUNBUFFERED": "1",
        "MAX_JOBS": str(config["correctness"]["max_jobs"]),
        "FLASHINFER_NVCC_THREADS": str(config["correctness"]["nvcc_threads"]),
    }
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    model = [config["model"], "--dtype", config["dtype"]]
    current = "nvidia-smi"

    def run(name, args):
        nonlocal current
        current = name
        run_logged(name, args, output, {**env, "RESULT_DIR": str(output / name)})

    def python(name, module, args):
        run(name, [sys.executable, "-m", module, *args])

    try:
        run("nvidia-smi", ["nvidia-smi"])
        # Keep long-context stress opt-in to avoid exhausting smaller GPUs in 'all'.
        stages = (
            ("check", "bench", "tail", "capacity")
            if config["stage"] == "all"
            else (config["stage"],)
        )
        for stage in stages:
            if stage == "check":
                check = config["correctness"]
                common = [
                    *model,
                    "--max-new-tokens",
                    str(check["max_new_tokens"]),
                    "--max-model-len",
                    str(check["max_model_len"]),
                ]
                for suite in ("unit", "integration"):
                    python(
                        f"{suite}-tests",
                        "pytest",
                        [
                            f"tests/{suite}",
                            "-x",
                            "-vv",
                            "-s",
                            "-ra",
                            "--tb=short",
                        ],
                    )
                python(
                    "hf-parity",
                    "scripts.verify_qwen3_5",
                    [
                        *common,
                        "--prompt",
                        check["prompt"],
                        "--result-file",
                        str(output / "hf-parity.json"),
                    ],
                )
                python(
                    "prefix-parity",
                    "scripts.verify_prefix_resume",
                    [
                        *common,
                        "--prefix-block-size",
                        str(check["prefix_block_size"]),
                        "--result-file",
                        str(output / "prefix-parity.json"),
                    ],
                )
            elif stage in ("bench", "tail", "long"):
                mode = config["benchmark"]["mode"] if stage == "bench" else stage
                run(stage, ["bash", "benchmarks/run.sh", mode])
            elif stage == "capacity":
                args = [*model, "--out-dir", str(output / "capacity")]
                for key, value in config["capacity"].items():
                    args.extend(["--" + key.replace("_", "-"), str(value)])
                python("capacity", "benchmarks.capacity_search", args)
            else:
                raise ValueError(f"Unknown stage: {stage}")
    except subprocess.CalledProcessError as error:
        code = error.returncode if error.returncode > 0 else 128 - error.returncode
        (output / "status.txt").write_text(f"FAILED stage={current} exit={code}\n")
        return code
    (output / "status.txt").write_text("COMPLETED\n")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tests.toml")
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args(argv)
    config = load_config(args.config, stage=args.stage)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(config["output_dir"]) / f"run_{stamp}"
    output.mkdir(parents=True)
    (output / "config.toml").write_bytes(args.config.read_bytes())
    (output / "effective_stage.txt").write_text(config["stage"] + "\n", encoding="utf-8")
    print(f"Results: {output}", flush=True)
    return execute(config, output)


if __name__ == "__main__":
    raise SystemExit(main())
