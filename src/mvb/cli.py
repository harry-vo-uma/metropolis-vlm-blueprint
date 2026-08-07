"""`mvb` command line.

Every subcommand maps to something in the quickstart, in the same order a
developer meets it: check the environment, build data, evaluate, inspect tools,
ask a question, serve. The `doctor` command exists because the developer-testing
rounds showed most first-run failures were environmental, and were being
diagnosed by reading tracebacks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Answer the questions that caused the most first-run failures."""
    from .config import get_settings
    from .nim.client import get_backend

    cfg = get_settings()
    checks: list[tuple[str, bool, str]] = []

    checks.append(("python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0]))

    suite = Path("eval/datasets/suite.jsonl")
    checks.append(("evaluation suite present", suite.exists(), str(suite) if suite.exists() else "run `make data`"))

    backend = get_backend()
    checks.append(
        (
            "backend",
            True,
            f"{backend.name} ({'live NIM' if cfg.nim.enabled else 'mock -- set NVIDIA_API_KEY for live'})",
        )
    )

    try:
        import fastapi  # noqa: F401

        checks.append(("fastapi installed", True, "ok"))
    except ImportError:
        checks.append(("fastapi installed", False, "pip install -e ."))

    try:
        import mcp  # noqa: F401

        checks.append(("mcp sdk (optional)", True, "ok"))
    except ImportError:
        checks.append(("mcp sdk (optional)", True, "not installed -- stdio server unavailable, JSON-RPC handler still works"))

    try:
        import peft  # noqa: F401
        import torch  # noqa: F401

        checks.append(("training extra (optional)", True, "ok"))
    except ImportError:
        checks.append(("training extra (optional)", True, "not installed -- eval and serving do not need it"))

    from .tools import registry

    checks.append(("tools registered", bool(registry.names()), ", ".join(registry.names())))

    width = max(len(name) for name, _, _ in checks)
    failed = 0
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{mark}] {name:<{width}}  {detail}")

    if failed:
        print(f"\n{failed} check(s) failed. See docs/troubleshooting.md")
    return 1 if failed else 0


def _cmd_data(args: argparse.Namespace) -> int:
    from .data.curate import curate
    from .data.synth import generate_pool
    from .evalsuite.suite import write_suite
    from .schemas import Provenance, Split

    pool = generate_pool(n=args.n, seed=args.seed)
    exclude = {Provenance.SYNTHETIC_AUGMENTED} if args.drop_synthetic else set()
    rows, report = curate(pool, exclude_provenance=exclude, seed=args.seed, target_size=args.size)

    out = Path(args.out)
    write_suite(out / "suite.jsonl", rows)
    for split in Split:
        write_suite(out / f"{split.value}.jsonl", [r for r in rows if r.split is split])
    print("\n".join(report.lines()))
    print(f"\nwrote {len(rows)} examples to {out}/suite.jsonl")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evalsuite.suite import load_suite, run_suite

    examples = load_suite(args.suite)
    for adapter in args.adapters:
        report, _, _ = run_suite(examples, adapter=adapter)
        print(report.summary_line())
        if args.by_task:
            for name, s in report.by_task.items():
                print(f"    {name:<24}n={s.n:<5}acc={s.accuracy:.3f}")
    return 0


def _cmd_tools(args: argparse.Namespace) -> int:
    from .tools import registry

    if args.call:
        arguments = json.loads(args.arguments) if args.arguments else {}
        result = registry.call(args.call, arguments)
        print(json.dumps(result.model_dump(), indent=2, default=str))
        return 0 if result.ok else 1
    print(json.dumps(registry.to_mcp_tools(), indent=2))
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from .serve.agent import run_agent

    run = run_agent(args.question, max_steps=args.max_steps)
    print(f"\nQ: {run.question}\nA: {run.answer}\n")
    print(f"tools used: {run.tool_names() or 'none'}")
    if run.truncated:
        print("NOTE: step budget exhausted; answer is partial")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("mvb.serve.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    from .mcpserver import client_config, serve_stdio

    if args.print_config:
        print(json.dumps(client_config(cwd=str(Path.cwd())), indent=2))
        return 0
    serve_stdio()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mvb", description="Metropolis VLM Blueprint")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check the environment and report what is missing")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("data", help="generate and curate the evaluation suite")
    p.add_argument("--n", type=int, default=1900)
    p.add_argument("--size", type=int, default=1500)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", default="eval/datasets")
    p.add_argument("--drop-synthetic", action="store_true")
    p.set_defaults(func=_cmd_data)

    p = sub.add_parser("eval", help="score adapters against the suite")
    p.add_argument("--suite", default="eval/datasets/suite.jsonl")
    p.add_argument("--adapters", nargs="*", default=["base", "lora-v3"])
    p.add_argument("--by-task", action="store_true")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("tools", help="list or invoke registered tools")
    p.add_argument("--call", help="tool name to invoke")
    p.add_argument("--arguments", help="JSON object of arguments")
    p.set_defaults(func=_cmd_tools)

    p = sub.add_parser("ask", help="run the agent loop against a question")
    p.add_argument("question")
    p.add_argument("--max-steps", type=int, default=6)
    p.set_defaults(func=_cmd_ask)

    p = sub.add_parser("serve", help="run the FastAPI service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("mcp", help="run the MCP stdio server")
    p.add_argument("--print-config", action="store_true", help="print an MCP client config stanza")
    p.set_defaults(func=_cmd_mcp)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
