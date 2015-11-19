#include "matrix.hpp"

// void fast_matrix_subtract (const matrix<float>& A, matrix<float>& C) {
// 	// assert(A.rows() == C.rows() && A.cols() == C.cols());
// 	size_t M = A.rows();
// 	size_t N = C.cols(), K = A.cols();
// 	float alpha = 1.0, beta = 1.0;
// 	int lda = M, ldb = 0, ldc = C.rows();
//
// 	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
// 							M, N, K,
// 							alpha, A.ptr(), lda,
// 							NULL, ldb,
// 							beta, C.mutable_ptr(), ldc);
// }
//
// void fast_matrix_subtract (const matrix<double>& A, matrix<double>& C) {
// 	// assert(A.rows() == C.rows() && A.cols() == C.cols());
// 	size_t M = A.rows();
// 	size_t N = C.cols(), K = A.cols();
// 	float alpha = 1.0, beta = 1.0;
// 	int lda = M, ldb = 0, ldc = C.rows();
//
// 	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
// 							M, N, K,
// 							alpha, A.ptr(), lda,
// 							NULL, ldb,
// 							beta, C.mutable_ptr(), ldc);
// }
//
// void fast_matrix_add (const matrix<float>& A, matrix<float>& C) {
// 	// assert(A.rows() == C.rows() && A.cols() == C.cols());
// 	size_t M = A.rows();
// 	size_t N = C.cols(), K = C.cols();
// 	float alpha = 1.0, beta = -1.0;
// 	int lda = M, ldb = 0, ldc = A.rows();
//
// 	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
// 							M, N, K,
// 							alpha, A.ptr(), lda,
// 							NULL, ldb,
// 							beta, C.mutable_ptr(), ldc);
// }
//
// void fast_matrix_add (const matrix<double>& A, matrix<double>& C) {
// 	// assert(A.rows() == C.rows() && A.cols() == C.cols());
// 	size_t M = A.rows();
// 	size_t N = C.cols(), K = C.cols();
// 	float alpha = 1.0, beta = -1.0;
// 	int lda = M, ldb = 0, ldc = A.rows();
//
// 	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
// 							M, N, K,
// 							alpha, A.ptr(), lda,
// 							NULL, ldb,
// 							beta, C.mutable_ptr(), ldc);
// }

matrix<float>& blas_pairwise_distance (const matrix<float>& A, const matrix<float>& B) {
	assert(A.cols() == B.cols());

	int M = A.rows();
	int N = B.rows();
	int K = B.cols();
	matrix<float> *C = new matrix<float>(M, N);
  float alpha = 1.0, beta = 0.0;
  int lda = A.cols(), ldb = B.cols(), ldc = C->cols();

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
							M, N, K,
							alpha, A.ptr(), lda,
							B.ptr(), ldb,
							beta, C->mutable_ptr(), ldc);
	float *ptr = C->mutable_ptr();

	const float *ptr_a = A.ptr();
	const float *ptr_b = B.ptr();

  #pragma omp parallel for
  for (size_t i=0; i < M; ++i) {
    for (size_t j=0; j < N; ++j) {
			for (size_t k=0; k < K; ++k)
				ptr[i*N + j] += ptr_a[i*K + k]*ptr_a[i*K + k] + ptr_b[j*K + k]*ptr_b[j*K + k];
			ptr[i*N + j] = sqrt(ptr[i*N + j]);
		}
	}
	return *C;
}

matrix<float>& blas_matrix_dot (const matrix<float>& A, const matrix<float>& B) {
  assert(A.cols() == B.rows());
	int M = A.rows();
	int N = B.cols();
	int K = B.rows();
	matrix<float> *C = new matrix<float>(A.rows(), B.cols());
  float alpha = 1.0, beta = 0.0;
  int lda = A.cols(), ldb = B.cols(), ldc = C->cols();

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
							M, N, K,
							alpha, A.ptr(), lda,
							B.ptr(), ldb,
							beta, C->mutable_ptr(), ldc);
  return *C;
}

matrix<double>& blas_matrix_dot (const matrix<double>& A, const matrix<double>& B) {
  assert(A.cols() == B.rows());
	int M = A.rows();
	int N = B.cols();
	int K = B.rows();
	matrix<double> *C = new matrix<double>(A.rows(), B.cols());
  float alpha = 1.0, beta = 0.0;
  int lda = A.cols(), ldb = B.rows(), ldc = C->cols();

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
							M, N, K,
							alpha, A.ptr(), lda,
							B.ptr(), ldb,
							beta, C->mutable_ptr(), ldc);
  return *C;
}
