"""Launcher that adds CUDA DLL directories before importing onnxruntime."""
import os
import sys
from pathlib import Path

# Add nvidia pip package DLL directories so onnxruntime can find CUDA libs
venv_site = Path(sys.executable).parent.parent / "Lib" / "site-packages" / "nvidia"
if venv_site.is_dir():
    for sub in venv_site.iterdir():
        bin_dir = sub / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

# Now run stereo_infer.main()
from stereo_infer import main
main()
