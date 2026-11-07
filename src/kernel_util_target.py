#!/usr/bin/env python3
"""Run under hcTracer, preserving the actual command's exit status."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    status_path = Path(sys.argv[1])
    env = dict(os.environ)
    hook = env.pop("GLU_NUMERIC_UTIL_HOOK")
    env["LD_PRELOAD"] = hook + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    try:
        child = subprocess.Popen(sys.argv[2:], env=env)
        status = {"pid": child.pid, "returncode": child.wait()}
    except OSError as error:
        status = {"returncode": 127, "error": str(error)}
    status_path.write_text(json.dumps(status) + "\n")
    code = status["returncode"]
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main())
