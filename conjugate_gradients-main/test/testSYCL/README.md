# FPGA Compilation Guide
## Prerequisites

Before initiating the compilation process, ensure that:

- You have an account on the system.
- You have access to the project number `p200301`.

## Step 1: Allocate Resources

Start by allocating the necessary resources with the `salloc` command, which grants access to an FPGA node for compilation.

    salloc -A p200301 -t 01:00:00 -q default -p fpga -N 1

- A p200301: Specifies the project account number.
- t 01:00:00: Sets the time allocation for the node (1 hour).
- q default: Chooses the default queue.
- p fpga: Specifies the FPGA partition.
- N 1: Requests one node.

## Step 2: Load Necessary Modules

After securing a node, load the required FPGA SDK and additional modules to prepare for compilation.

    module load ifpgasdk 520nmx intel env/release/latest ifpga/2021.3.0

### Step 2a: Load MKL Library

Additionally, load the MKL library for optimized mathematical computations.

    module load env/release/2023.1 imkl/2023.1.0

## Step 3: Compilation Variants

Choose your compilation type based on the test objectives, with detailed commands for each scenario, including the integration with the MKL library.

### Step 3a: Emulation

Emulation tests, validates, and debugs FPGA designs before hardware deployment. It's not performance-focused but crucial for initial validation.

    module load intel-compilers

**Standard Compilation:**

    icpx -fsycl -fintelfpga main.cpp -o main.fpga_emu |& tee compilation.log

- `icpx`: Intel's DPC++/C++ Compiler.
- `-fsycl`: Enables SYCL language support.
- `-fintelfpga`: Enables FPGA optimizations and extensions for Intel devices.
- `main.cpp`: Source file to compile.
- `-o main.fpga_emu`: Specifies the output file for the compiled program.
- `|& tee compilation.log`: Pipes both stdout and stderr to the `tee` command, which logs output to `compilation.log` and displays it on the screen.

**With MKL Library:**

    icpx -fsycl -DMKL_ILP64 -I${MKLROOT}/include main.cpp -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -ltbb -pthread -ldl -lm -o main.fpga_emu

### Step 3b: Static Reports

Generate optimization reports to analyze design efficiency. After reaching the FPGA image object checkpoint, a finished FPGA image is produced.

**Standard Compilation:**

    icpx -fsycl -Xshardware -fsycl-link=early -Xstarget=Stratix10 main.cpp -o main_report.a

- `icpx`: Intel's DPC++/C++ Compiler.
- `-fsycl`: Enables SYCL language support.
- `-Xshardware`: Targets the generation of FPGA hardware instead of emulation.
- `-fsycl-link=early`: Generates an intermediate FPGA binary that can be used for early testing or for generating reports.
- `-Xstarget=Stratix10`: Specifies the FPGA device target, in this case, Stratix 10.
- `main.cpp`: Source file to compile.
- `-o main_report.a`: Specifies the output file for the compiled report.

**With MKL Library:**

    icpx -fsycl -DMKL_ILP64 -I${MKLROOT}/include -Xshardware -fsycl-link=early -Xstarget=Stratix10 main.cpp -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -ltbb -pthread -ldl -lm -o main_report.a

### Step 3c: Full Compilation

This step involves several hours of compilation. Pre-validation through emulation is recommended. Using -Xsfast-compile decreases compile time at the expense of final performance.

**Standard Compilation:**

    icpx -fsycl -Xshardware -Xstarget=Stratix10 -DFPGA_HARDWARE main.cpp -o main.fpga

- `icpx`: Intel's DPC++/C++ Compiler.
- `-fsycl`: Enables SYCL language support.
- `-Xshardware`: Targets the generation of FPGA hardware instead of emulation.
- `-Xstarget=Stratix10`: Specifies the FPGA device target, in this case, Stratix 10.
- `-DFPGA_HARDWARE`: Defines a preprocessor macro `FPGA_HARDWARE`, which can be used to conditionally compile code specific for the FPGA hardware target.
- `main.cpp`: Source file to compile.
- `-o main.fpga`: Specifies the output file for the fully compiled FPGA binary.

**With MKL Library:**

    icpx -fsycl -DMKL_ILP64 -I${MKLROOT}/include -Xshardware -Xstarget=Stratix10 -DFPGA_HARDWARE main.cpp -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -ltbb -pthread -ldl -lm -o main.fpga

## Additional Information

For more detailed guidance and resources on FPGA compilation using oneAPI, refer to https://ekieffer.github.io/oneAPI-FPGA/.
