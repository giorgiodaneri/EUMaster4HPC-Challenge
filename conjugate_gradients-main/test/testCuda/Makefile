compile:
	g++ -c -O2 mainCuda.cpp
	nvcc -arch=sm_80 -c -O3 CGSolverCUDA.cu -lcublas
	nvcc -o main mainCuda.o CGSolverCUDA.o -lcublas -O3 -use_fast_math -Xptxas -O3 -Xlinker -O3
