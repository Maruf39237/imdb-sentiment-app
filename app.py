import sys

import spaces


# Register a ZeroGPU-compatible function.
@spaces.GPU
def _zerogpu_compat():
    return None


# Manually send the ZeroGPU startup report.
# This is needed because we are launching Streamlit directly
# instead of calling the normal Gradio launch() function.
try:
    from spaces.zero import startup as _zero_startup

    _zero_startup()
    print("[zerogpu] startup report sent", flush=True)

except ImportError:
    print("[zerogpu] spaces.zero.startup not available", flush=True)

except Exception as exc:
    print(f"[zerogpu] startup report failed: {exc}", flush=True)


# Start the existing Streamlit application.
os_exec = sys.executable

import os

os.execv(
    os_exec,
    [
        os_exec,
        "-m",
        "streamlit",
        "run",
        "main.py",
        "--server.address=0.0.0.0",
        "--server.port=7860",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ],
)