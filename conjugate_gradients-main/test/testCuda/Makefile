# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -Ofast -funroll-loops -march=native
NVCCFLAGS = -gen-opt-lto -maxrregcount=12 -dlto -arch=sm_80 -O3 -use_fast_math -Xptxas -O3 -Xlinker -O3 -Xcompiler -Ofast 

# Libraries
LIBS = -lcublas

# Targets
TARGET = main
CUDA_OBJ = mainCuda.o CGSolverCUDA.o

# Make rules
all: $(TARGET)

$(TARGET): $(CUDA_OBJ)
	$(NVCC) -o $@ $^ $(LIBS) $(NVCCFLAGS)

mainCuda.o: mainCuda.cpp
	$(CXX) -c $< $(CXXFLAGS)

CGSolverCUDA.o: CGSolverCUDA.cu
	$(NVCC) -c $< $(NVCCFLAGS)

clean:
	rm -f $(TARGET) $(CUDA_OBJ)

