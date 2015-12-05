#include "matrix.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

void cuda_init (int& dev_id) {
  cudaError_t error;
  dev_id = 0;
  error = cudaGetDevice(&dev_id);
  if (error != cudaSuccess) {
    cout << "cudaGetDevice returned error code " << error << ", line " << __LINE__;
    exit(EXIT_FAILURE);
  }
  //
  cudaDeviceProp device_prop;
  error = cudaGetDeviceProperties(&device_prop, dev_id);
  if (error != cudaSuccess) {
    cout << "cudaGetDeviceProperties returned error code " << error << ", line " << __LINE__;
    exit(EXIT_FAILURE);
  }
  cout << "GPU device " << dev_id << " " << device_prop.name << " with compute capability " << device_prop.major << "." << device_prop.minor << endl;
}

matrix<float> cuda_matrix_dot (const matrix<float>& A, const matrix<float>& B) {
  assert(A.cols() == B.rows());
  int M = A.rows();
	int N = B.cols();
	int K = B.rows();
	matrix<float> C(M, N);

  const float *ptr_a = A.ptr();
  const float *ptr_b = B.ptr();
  float *ptr_c = C.mutable_ptr();

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasHandle_t handle;
  cublasCreate(&handle);
  // device memory
  float *ptr_A, *ptr_B, *ptr_C;
  cudaMalloc((void**) &ptr_A, sizeof(float)*A.size());
  cudaMalloc((void**) &ptr_B, sizeof(float)*B.size());
  cudaMalloc((void**) &ptr_C, sizeof(float)*C.size());
  cudaMemcpy(ptr_A, ptr_a, sizeof(float)*A.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(ptr_B, ptr_b, sizeof(float)*B.size(), cudaMemcpyHostToDevice);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, ptr_B, N, ptr_A, K, &beta, ptr_C, N);
  cudaMemcpy(ptr_c, ptr_C, sizeof(float)*C.size(), cudaMemcpyDeviceToHost);
  cudaFree(ptr_A);
  cudaFree(ptr_B);
  cudaFree(ptr_C);
  cublasDestroy(handle);
  return C;
}

struct abs2 {
  __host__ __device__ double operator() (const float &x) const { return x*x; }
};

__global__ void assemble_final_result (const float * __restrict__ d_norms_x_2, const float * __restrict__ d_norms_y_2, float * __restrict__ d_dots, const int num_rows, const int num_cols) {
  const int j = threadIdx.x + blockIdx.x * gridDim.x;
	const int i = threadIdx.y + blockIdx.y * gridDim.y;
  if ((j < num_cols) && (i < num_rows))
    d_dots[i * num_cols + j] = d_norms_x_2[i] + d_norms_y_2[j] - 2 * d_dots[i * num_cols + j];
}

int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

matrix<float> cuda_pairwise_distance (const matrix<float>& A, const matrix<float>& B) {
  assert(A.cols() == B.cols());

  // pointer references to host memory variables
  thrust::device_vector<float> d_A(A.ptr(), A.ptr() + A.size());
  thrust::device_vector<float> d_B(B.ptr(), B.ptr() + B.size());
  float *ptr_A = thrust::raw_pointer_cast(d_A.data());
  float *ptr_B = thrust::raw_pointer_cast(d_B.data());

  float *ptr_C;
  cudaMalloc((void**) &ptr_C, sizeof(float)*A.rows()*B.rows());

  cublasHandle_t handle;
  cublasCreate(&handle);

  // compute the scalar product
  const float alpha = 1.0f;
  const float beta = 0.0;
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, B.rows(), A.rows(), A.cols(), &alpha, ptr_B, B.cols(), ptr_A, A.cols(), &beta, ptr_C, B.rows());

  // compute L2-norm of A
  thrust::device_vector<float> d_A_2(A.size());
  thrust::transform(d_A.begin(), d_A.end(), d_A_2.begin(), abs2()); // a^2
  thrust::device_vector<float> d_norms_A_2 (A.rows());
  thrust::device_vector<float> d_ones(A.cols(), 1.f);
  cublasSgemv(handle, CUBLAS_OP_T, A.cols(), A.rows(), &alpha, thrust::raw_pointer_cast(d_A_2.data()), A.cols(), thrust::raw_pointer_cast(d_ones.data()), 1, &beta, thrust::raw_pointer_cast(d_norms_A_2.data()), 1); // sum(a^2)

  // compute L2-norm of B
  thrust::device_vector<float> d_B_2(B.size());
  thrust::transform(d_B.begin(), d_B.end(), d_B_2.begin(), abs2()); // b^2
  thrust::device_vector<float> d_norms_B_2(B.rows());
  cublasSgemv(handle, CUBLAS_OP_T, B.cols(), B.rows(), &alpha, thrust::raw_pointer_cast(d_B_2.data()), B.cols(), thrust::raw_pointer_cast(d_ones.data()), 1, &beta, thrust::raw_pointer_cast(d_norms_B_2.data()), 1); // sum(B^2)

  // assemble the final result
  dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 dimGrid(iDivUp(A.rows(), BLOCK_SIZE_X), iDivUp(B.rows(), BLOCK_SIZE_Y));
  assemble_final_result<<<dimGrid, dimBlock>>> (thrust::raw_pointer_cast(d_norms_A_2.data()), thrust::raw_pointer_cast(d_norms_B_2.data()), ptr_C, A.rows(), B.rows());

  // copy result from device to host
  matrix<float> C(A.rows(), B.rows());
  float *ptr_c = C.mutable_ptr();
  cudaMemcpy(ptr_c, ptr_C, sizeof(float)*C.size(), cudaMemcpyDeviceToHost);
  cudaFree(ptr_C);
  cublasDestroy(handle);
  return C;
}

matrix<float> cuda_pairwise_distance (const matrix<float>& A) {
  // pointer references to host memory variables
  thrust::device_vector<float> d_A(A.ptr(), A.ptr() + A.size());
  float *ptr_A = thrust::raw_pointer_cast(d_A.data());

  float *ptr_C;
  cudaMalloc((void**) &ptr_C, sizeof(float)*A.rows()*A.rows());

  cublasHandle_t handle;
  cublasCreate(&handle);

  // compute the scalar product
  const float alpha = 1.0f;
  const float beta = 0.0;
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, A.rows(), A.rows(), A.cols(), &alpha, ptr_A, A.cols(), ptr_A, A.cols(), &beta, ptr_C, A.rows());

  // compute L2-norm of A
  thrust::device_vector<float> d_A_2(A.size());
  thrust::transform(d_A.begin(), d_A.end(), d_A_2.begin(), abs2()); // a^2
  thrust::device_vector<float> d_norms_A_2 (A.rows());
  thrust::device_vector<float> d_ones(A.cols(), 1.f);
  cublasSgemv(handle, CUBLAS_OP_T, A.cols(), A.rows(), &alpha, thrust::raw_pointer_cast(d_A_2.data()), A.cols(), thrust::raw_pointer_cast(d_ones.data()), 1, &beta, thrust::raw_pointer_cast(d_norms_A_2.data()), 1); // sum(a^2)

  // assemble the final result
  dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 dimGrid(iDivUp(A.rows(), BLOCK_SIZE_X), iDivUp(A.rows(), BLOCK_SIZE_Y));
  assemble_final_result<<<dimGrid, dimBlock>>> (thrust::raw_pointer_cast(d_norms_A_2.data()), thrust::raw_pointer_cast(d_norms_A_2.data()), ptr_C, A.rows(), A.rows());

  // copy result from device to host
  matrix<float> C(A.rows(), A.rows());
  float *ptr_c = C.mutable_ptr();
  cudaMemcpy(ptr_c, ptr_C, sizeof(float)*C.size(), cudaMemcpyDeviceToHost);
  cudaFree(ptr_C);
  cublasDestroy(handle);
  return C;
}
