#!/usr/bin/env python3
import subprocess
import time
import statistics
import sys

def measure_time(command_with_args, num_samples):
    execution_times = []
    print("")
    for i in range(num_samples):
        start_time = time.time()
        completed_process = subprocess.run(command_with_args,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        end_time = time.time()

        exec_time = end_time - start_time
        print(f"Sample {i}: took {exec_time} seconds")
        print("Output:", completed_process.stdout)
        print("Error:\n", completed_process.stderr)
        execution_times.append(exec_time)
    print("")
    return execution_times

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: {} <num_samples> <command>".format(sys.argv[0]))
        sys.exit(1)

    num_samples = int(sys.argv[1])
    command_with_args = sys.argv[2:]

    execution_times = measure_time(command_with_args, num_samples)

    t_min = min(execution_times)
    t_max = max(execution_times)
    t_avg = statistics.mean(execution_times) if num_samples > 1 else t_min
    t_dev = statistics.stdev(execution_times) if num_samples > 1 else 0

    print(f"Min time:     {t_min} seconds")
    print(f"Max time:     {t_max} seconds")
    print(f"Average time: {t_avg} seconds")
    print(f"Std dev:      {t_dev} seconds")