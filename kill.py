import carla
import os
import signal
import psutil
import subprocess
import time

processes = [p for p in psutil.process_iter() if "carla" in p.name().lower()]
for process in processes:
    print(process.pid)
    os.kill(process.pid, signal.SIGKILL)


