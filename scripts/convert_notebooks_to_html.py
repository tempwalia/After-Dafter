#!/usr/bin/env python3

from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import nbformat
from nbconvert import HTMLExporter

try:
    import papermill as pm  # optional, only required when --execute
except Exception:
    pm = None  # handled at runtime if --execute is requested


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert all notebooks to HTML and save in ml_models")
    p.add_argument("--notebooks-dir", type=str, default=None, help="Directory containing .ipynb notebooks")
    p.add_argument("--out-dir", type=str, default=None, help="Directory to write HTML files")
    p.add_argument("--execute", action="store_true", help="Execute notebooks before converting")
    p.add_argument("--param", action="append", default=[], help="Notebook parameters as key=value (repeatable)")
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    # scripts/convert_notebooks_to_html.py => repo_root is parent of scripts
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    notebooks_dir = Path(args.notebooks_dir) if args.notebooks_dir else (repo_root / "notebooks")
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "ml_models")
    return repo_root, notebooks_dir, out_dir


def parse_params(param_list: List[str]) -> Dict[str, object]:
    params: Dict[str, object] = {}
    for item in param_list:
        if "=" not in item:
            print(f"Warning: skipping malformed --param '{item}', expected key=value")
            continue
        k, v = item.split("=", 1)
        # best-effort cast to numeric if possible
        v_cast: object
        try:
            if v.isdigit():
                v_cast = int(v)
            else:
                v_cast = float(v)
        except Exception:
            v_cast = v
        params[k] = v_cast
    return params


def sanitize_name(path: Path) -> str:
    # Create a safe filename for HTML output based on notebook stem
    name = path.stem.replace(" ", "_").replace("/", "_")
    return f"{name}.html"


def export_to_html(nb_path: Path, out_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with nb_path.open("r", encoding="utf-8") as f:
            nb_node = nbformat.read(f, as_version=4)
        html_exporter = HTMLExporter()
        html_exporter.template_name = "classic"
        body, _ = html_exporter.from_notebook_node(nb_node)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(body)


def execute_notebook(in_path: Path, executed_path: Path, params: Dict[str, object]) -> None:
    if pm is None:
        raise RuntimeError("papermill is not available; install it or run without --execute")
    executed_path.parent.mkdir(parents=True, exist_ok=True)
    pm.execute_notebook(
        input_path=str(in_path),
        output_path=str(executed_path),
        parameters=params or {},
        progress_bar=False,
        request_save_on_cell_execute=False,
        log_output=False,
    )


def main() -> int:
    args = parse_args()
    repo_root, notebooks_dir, out_dir = resolve_paths(args)
    params = parse_params(args.param)

    if not notebooks_dir.exists():
        print(f"Notebooks directory not found: {notebooks_dir}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect notebooks (exclude .ipynb_checkpoints)
    notebooks = [p for p in notebooks_dir.rglob("*.ipynb") if ".ipynb_checkpoints" not in str(p)]
    if not notebooks:
        print(f"No notebooks found under {notebooks_dir}")
        return 0

    print(f"Found {len(notebooks)} notebooks under {notebooks_dir}")
    print(f"Output directory: {out_dir}")
    if args.execute:
        print("Execution enabled: notebooks will be run before conversion")
        if pm is None:
            print("Error: papermill not installed; cannot execute notebooks. Install papermill or omit --execute.")
            return 2
        if params:
            print(f"Parameters: {params}")

    successes, failures = 0, 0
    for nb in sorted(notebooks):
        try:
            html_name = sanitize_name(nb)
            out_html = out_dir / html_name
            print(f"\nProcessing: {nb.relative_to(repo_root)} -> {out_html.relative_to(repo_root)}")

            if args.execute:
                executed_nb = out_dir / (nb.stem.replace(" ", "_") + "__executed.ipynb")
                execute_notebook(nb, executed_nb, params)
                export_to_html(executed_nb, out_html)
                # Optionally remove executed notebook; keep for debugging
                # executed_nb.unlink(missing_ok=True)
            else:
                export_to_html(nb, out_html)

            print("OK")
            successes += 1
        except Exception as e:
            print(f"FAILED: {nb} -> {e}")
            failures += 1

    print(f"\nDone. Success: {successes}, Failed: {failures}, Output dir: {out_dir}")
    return 0 if failures == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
