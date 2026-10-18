"""Execute all four notebooks using this project's kernel, preserving outputs."""

import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from ipykernel.kernelspec import install

ROOT = Path(__file__).resolve().parents[1]
install(user=True, kernel_name="herovired", display_name="HeroVired (.venv)")
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    print("Executing", path.name, flush=True)
    nb = nbformat.read(path, as_version=4)
    NotebookClient(
        nb,
        timeout=1200,
        kernel_name="herovired",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    nbformat.write(nb, path)
    print("Saved executed outputs:", path.name, flush=True)
