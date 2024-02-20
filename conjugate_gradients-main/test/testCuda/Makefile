compile:
	g++ -c mainCuda.cpp
	nvcc -arch=sm_80 -c CGSolverCUDA.cu
	nvcc -o main mainCuda.o CGSolverCUDA.o
