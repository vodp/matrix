nvcc -c -arch=sm_20 matrix_cuda.cu
g++ -o csil matrix_blas.cpp compute_sil.cpp matrix_cuda.o -L/usr/local/cuda-6.5/lib64 -I/usr/local/cuda-6.5/include -lopenblas -lpthread -lcudart -lcuda -lcublas -fopenmp -O3 -Wextra -std=c++0x
