import subprocess
import time
import sys
import argparse

def parse_powermetrics(output):
    metrics = {}
    lines = output.split('\n')
    for line in lines:
        if line.startswith("CPU Power:"):
            metrics['CPU Power'] = line.split(":", 1)[1].strip()
        elif line.startswith("GPU Power:"):
            metrics['GPU Power'] = line.split(":", 1)[1].strip()
        elif line.startswith("ANE Power:"):
            metrics['ANE Power'] = line.split(":", 1)[1].strip()
        elif line.startswith("Combined Power"):
            metrics['Combined Power'] = line.split(":", 1)[1].strip()
        elif line.startswith("Thermal Pressure:"):
            metrics['Thermal Pressure'] = line.split(":", 1)[1].strip()
    return metrics

def run_powermetrics(interval_ms):
    print(f"Starting powermetrics monitor (interval: {interval_ms}ms)...")
    print("Note: This script requires sudo privileges to read powermetrics.")
    print("Press Ctrl+C to stop.\n")
    
    cmd = [
        "sudo", "powermetrics", 
        "-n", "1", 
        "-i", str(interval_ms), 
        "--samplers", "cpu_power,gpu_power,ane_power,thermal"
    ]
    
    try:
        while True:
            # powermetrics blocks for interval_ms while sampling
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            metrics = parse_powermetrics(result.stdout)
            
            # Clear previous lines or print separated blocks
            print("-" * 50)
            print(f"Time: {time.strftime('%H:%M:%S')}")
            print(f"Thermal Pressure : {metrics.get('Thermal Pressure', 'N/A')}")
            print(f"CPU Power        : {metrics.get('CPU Power', 'N/A')}")
            print(f"GPU Power        : {metrics.get('GPU Power', 'N/A')}")
            print(f"ANE Power        : {metrics.get('ANE Power', 'N/A')}")
            print(f"Combined Power   : {metrics.get('Combined Power', 'N/A')}")
            
    except subprocess.CalledProcessError as e:
        print(f"\nError running powermetrics: {e}")
        print("Did you enter your password correctly? Powermetrics requires sudo.")
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor Mac Power Metrics")
    parser.add_argument("--interval", type=int, default=1000, help="Sampling interval in ms (default: 1000)")
    args = parser.parse_args()
    
    run_powermetrics(args.interval)
