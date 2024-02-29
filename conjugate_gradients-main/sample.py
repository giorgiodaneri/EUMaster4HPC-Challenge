#!/usr/bin/env python3
import subprocess
import statistics
import sys
import re

def measure_time(command_with_args, num_samples):
    execution_times = []
    print("")
    for i in range(num_samples):
        completed_process = subprocess.run(command_with_args,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        output = completed_process.stdout
        
        # Extracting time taken from the output using regular expression
        match = re.search(r"Time taken by function: (\d+) milliseconds", output)
        if match:
            time_taken = int(match.group(1)) / 1000  # Converting milliseconds to seconds
            print(f"Sample {i}: took {time_taken} seconds")
            execution_times.append(time_taken)
        else:
            print(f"Sample {i}: Time taken not found in output")

        print("Output:", output)
        print("Error:\n", completed_process.stderr)
    print("")
    return execution_times

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: {} <num_samples> <command>".format(sys.argv[0]))
        sys.exit(1)

    num_samples = int(sys.argv[1])
    command_with_args = sys.argv[2:]

    execution_times = measure_time(command_with_args, num_samples)

    t_min = min(execution_times) if execution_times else 0
    t_max = max(execution_times) if execution_times else 0
    t_avg = statistics.mean(execution_times) if execution_times else 0
    t_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0

    print(f"Min time:     {t_min} seconds")
    print(f"Max time:     {t_max} seconds")
    print(f"Average time: {t_avg} seconds")
    print(f"Std dev:      {t_dev} seconds")
