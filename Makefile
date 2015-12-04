all: matrix_cuda.o matrixview.hpp compute_csil.cpp
	g++ -o csil matrix_blas.cpp alg.cpp compute_sil.cpp matrix_cuda.o -L/usr/local/cuda-7.0/lib64 -I/usr/local/cuda-7.0/include -lopenblas -lpthread -lcudart -lcuda -lcublas -fopenmp -O3 -Wextra -std=c++11
matrix_cuda.o: marix_cuda.cu matrix.hpp
	nvcc -std=c++11 -c -arch=sm_20 matrix_cuda.cu
clean:
	rm -rf csil
