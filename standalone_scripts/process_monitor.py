# save as process_monitor.py
# run alongside profiler to get model-only CPU stats

import psutil
import time
import csv
import sys
import subprocess
import re
from datetime import datetime

# find your model process name
PROCESS_NAME = "python3"   # change to your exact script name
OUTPUT_CSV   = "./power_logs/process_cpu_log.csv"

def get_system_power_mw():
    # Runs powermetrics for a short 100ms interval to grab the current power snapshot
    try:
        cmd = ["sudo", "powermetrics", "-n", "1", "-i", "100", "--samplers", "cpu_power,ane_power"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        cpu_match = re.search(r"CPU Power:\s+([\d.]+)\s+mW", res.stdout)
        ane_match = re.search(r"ANE Power:\s+([\d.]+)\s+mW", res.stdout)
        
        cpu_mw = float(cpu_match.group(1)) if cpu_match else 0.0
        ane_mw = float(ane_match.group(1)) if ane_match else 0.0
        return cpu_mw, ane_mw
    except Exception:
        return 0.0, 0.0

def find_model_process():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = " ".join(proc.info['cmdline'] or [])
            # Search for any llama script (e.g. run_llama_3_2_1b.py or run_llama_3_2_3b.py)
            if "run_llama_3_2" in cmdline and "process_monitor" not in cmdline:
                return proc
        except:
            pass
    return None

def monitor(duration=60):
    print("Installing psutil if needed...")
    
    rows = []
    start = time.time()
    
    print("Searching for model process...")
    proc = None
    
    while time.time() - start < duration:
        if proc is None:
            proc = find_model_process()
            if proc:
                print(f"Found process PID {proc.pid}")
            else:
                print("Waiting for model process...")
                time.sleep(2)
                continue

        try:
            # cpu_percent over 1 second interval
            cpu_pct  = proc.cpu_percent(interval=1.0)
            mem_mb   = proc.memory_info().rss / 1024 / 1024
            
            # Fetch the system power via powermetrics right after
            sys_cpu_mw, sys_ane_mw = get_system_power_mw()
            
            elapsed  = round(time.time() - start, 1)

            row = {
                "timestamp":  datetime.now().isoformat(),
                "elapsed":    elapsed,
                "pid":        proc.pid,
                "cpu_pct":    cpu_pct,
                "mem_mb":     round(mem_mb, 1),
                "sys_cpu_mw": sys_cpu_mw,
                "sys_ane_mw": sys_ane_mw,
            }
            rows.append(row)
            print(f"[{elapsed:>6.1f}s]  PID CPU: {cpu_pct:>6.1f}%  RAM: {mem_mb:>7.1f} MB  |  Sys CPU Power: {sys_cpu_mw:>6.1f} mW  Sys ANE Power: {sys_ane_mw:>6.1f} mW")

        except psutil.NoSuchProcess:
            print("Process ended.")
            break

    # save CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp","elapsed","pid","cpu_pct","mem_mb","sys_cpu_mw","sys_ane_mw"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved → {OUTPUT_CSV}")

    # print summary
    if rows:
        avg_cpu = sum(r["cpu_pct"] for r in rows) / len(rows)
        avg_mem = sum(r["mem_mb"] for r in rows) / len(rows)
        print(f"Avg CPU: {avg_cpu:.1f}%  |  Avg RAM: {avg_mem:.1f} MB")

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    monitor(duration)