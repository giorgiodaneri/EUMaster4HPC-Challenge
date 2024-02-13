compile:
        icpx -Wall -O3 -march=native -qopenmp -qopenmp-simd -Iinclude/ src/conjugate_gradients.cpp test/mainOmp.cpp -o mainOmp

run:
        ./conjugate_gradients io/matrix.bin io/rhs.bin io/sol.bin