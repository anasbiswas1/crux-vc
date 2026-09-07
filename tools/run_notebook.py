#!/usr/bin/env python
"""Execute one notebook in place or to an explicit output path."""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    parser.add_argument("--output")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    source = Path(args.notebook).resolve()
    output = Path(args.output).resolve() if args.output else source.with_name(source.stem + ".executed.ipynb")
    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(notebook, timeout=args.timeout, kernel_name="python3", allow_errors=False)
    client.execute(cwd=str(source.parents[1]))
    nbformat.write(notebook, output)
    print(output)


if __name__ == "__main__":
    main()
