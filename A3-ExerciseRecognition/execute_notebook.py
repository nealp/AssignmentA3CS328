"""Run the assignment notebook with the Python environment running this script."""

import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


if __name__ == "__main__":
    notebook_path = Path(__file__).resolve().with_name("A3_exercise_recognition.ipynb")
    with tempfile.TemporaryDirectory(prefix="a3-notebook-") as runtime:
        os.environ["MPLCONFIGDIR"] = str(Path(runtime) / "matplotlib")
        os.environ["IPYTHONDIR"] = str(Path(runtime) / "ipython")
        os.environ["JUPYTER_RUNTIME_DIR"] = str(Path(runtime) / "jupyter")
        os.environ["JUPYTER_PATH"] = str(Path(sys.prefix) / "share" / "jupyter")
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(notebook_path.parent)}},
        )
        print("Executing notebook with the project environment...", flush=True)
        client.execute()
        nbformat.write(notebook, notebook_path)
        print("Notebook completed and saved with outputs.", flush=True)
