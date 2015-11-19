#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <math.h>
#include <cblas.h>
#include <assert.h>
#include <limits>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

template <class T>
class matrix
{
public:
		matrix ();
		matrix (size_t rows, size_t cols);
		matrix (T *raw_data, size_t rows, size_t cols);
		matrix (const matrix<T>& mx);
		matrix (matrix<T>&& mx);
		~matrix ();

		const T* ptr() const;
		T* mutable_ptr();

		// slicing, indexing operators
		T& operator() (size_t i, size_t j);
		T operator() (size_t i, size_t j) const;

		size_t rows() const;
		size_t cols() const;
		size_t size() const;
		void size(size_t rows, size_t cols);

		// arimethic operators
		matrix<T>& add(T t);
		matrix<T>& scale(T t);
		matrix<T>& power(T t);

		matrix<T>& t ();

		T max () const;
		T min () const;
		T sum () const;
		matrix<T>& max (int axis) const;
		matrix<T>& min (int axis) const;
		matrix<T>& sum (int axis) const;

		// fillers
		matrix<T>& fill (T value);
		matrix<T>& range (T v1, T v2);
		matrix<T>& randn ();

		// slicing methods
		view<T>& r_ (const vector<size_t>& rows);
		view<T>& r_ (const matrix<bool>& rows);
		view<T>& r_ (size_t r1, size_t r2);

		view<T>& c_ (const vector<size_t>& cols);
		view<T>& c_ (const matrix<bool>& cols);
		view<T>& c_ (size_t c1, size_t c2);

		view<T>& grid (const matrix<bool>& rows, const matrix<bool>& cols);
		view<T>& grid (const vector<size_t>& rows, const std::vector<size_t>& cols);
		view<T>& area (size_t r1, size_t r2, size_t c1, size_t c2);

		// assignment operators
		matrix<T>& operator= (matrix<T>&& mx);
		matrix<T>& operator= (const matrix<T>& mx);
		matrix<T>& operator= (const view<T>& mx);

		// element-wise operators
		matrix<T>& add(const matrix<T>& mx);
		matrix<T>& mul(const matrix<T>& mx);
		matrix<T>& add(const view<T>& mx);
		matrix<T>& mul(const view<T>& mx);

		// dot-product operator
		matrix<T>& dot(const matrix<T>& mx);
		matrix<T>& dot(const view<T>& mx);

		// static methods
		static matrix<T>* load (const char* filename, size_t rows, size_t cols);
		static void 			dump (const char* filename, matrix<T>& mx);

		static matrix<T>& tile (const matrix<T>& pattern, size_t nrows, size_t ncols);

private:
		size_t num_rows;
		size_t num_cols;
		vector<T> data;
};

template<class T> ostream& operator<< (ostream& os, const matrix<T>& mx);

template<class T> matrix<T>& operator+ (const matrix<T>& A, const matrix<T>& B);
template<class T> matrix<T>& operator+ (const matrix<T>& A, const T t);
template<class T> matrix<T>& operator+ (const T t, const matrix<T>& A);

template<class T> matrix<T>& operator- (const matrix<T>& A, const matrix<T>& B);
template<class T> matrix<T>& operator- (const matrix<T>& A, const T t);
template<class T> matrix<T>& operator- (const T t, const matrix<T>& A);

template<class T> matrix<T>& operator* (const matrix<T>& A, const matrix<T>& B);
template<class T> matrix<T>& operator* (const matrix<T>& A, const T t);
template<class T> matrix<T>& operator* (const T t, const matrix<T>& A);

template<class T> matrix<T>& operator/ (const matrix<T>& A, const matrix<T>& B);
template<class T> matrix<T>& operator/ (const matrix<T>& A, const T t);
template<class T> matrix<T>& operator/ (const T t, const matrix<T>& A);

template<class T> matrix<bool>&  operator< (const matrix<T>& mx, const matrix<T>& mx2);
template<class T> matrix<bool>&  operator<= (const matrix<T>& mx, const matrix<T>& mx2);
template<class T> matrix<bool>&  operator> (const matrix<T>& mx, const matrix<T>& mx2);
template<class T> matrix<bool>&  operator>= (const matrix<T>& mx, const matrix<T>& mx2);
template<class T> matrix<bool>&  operator== (const matrix<T>& mx, const matrix<T>& mx2);
template<class T> matrix<bool>&  operator!= (const matrix<T>& mx, const matrix<T>& mx2);

template<class T> matrix<bool>&  operator< (const matrix<T>& mx, const T t);
template<class T> matrix<bool>&  operator<= (const matrix<T>& mx, const T t);
template<class T> matrix<bool>&  operator> (const matrix<T>& mx, const T t);
template<class T> matrix<bool>&  operator>= (const matrix<T>& mx, const T t);
template<class T> matrix<bool>&  operator== (const matrix<T>& mx, const T t);
template<class T> matrix<bool>&  operator!= (const matrix<T>& mx, const T t);

template<class T> matrix<bool>&  operator< (const T t, const matrix<T>& mx);
template<class T> matrix<bool>&  operator<= (const T t, const matrix<T>& mx);
template<class T> matrix<bool>&  operator> (const T t, const matrix<T>& mx);
template<class T> matrix<bool>&  operator>= (const T t, const matrix<T>& mx);
template<class T> matrix<bool>&  operator== (const T t, const matrix<T>& mx);
template<class T> matrix<bool>&  operator!= (const T t, const matrix<T>& mx);

template<class T> matrix<T>& operator^ (const matrix<T>& A, const T t);

matrix<double>& blas_matrix_dot (const matrix<double>& A, const matrix<double>& B);
matrix<float>& blas_matrix_dot (const matrix<float>& A, const matrix<float>& B);
matrix<float>& blas_pairwise_distance (const matrix<float>& A, const matrix<float>& B);
matrix<double>& blas_pairwise_distance (const matrix<double>& A, const matrix<double>& B);

void 					cuda_init (int& dev_id);
matrix<float>& cuda_matrix_dot (const matrix<float>& A, const matrix<float>& B);
matrix<float>& cuda_pairwise_distance (const matrix<float>& A, const matrix<float>& B);
matrix<float>& cuda_pairwise_distance (const matrix<float>& A);
////////////////////////////////////////////////////////////////////////////////
template<class T> matrix<T>& matrix<T>::t () {
	T tmp;
	T *ptr = data.data();
	for (size_t i=0; i < num_rows; ++i)
		for (size_t j=0; j < num_cols; ++j) {
			tmp = ptr[i * num_cols + j];
			ptr[i * num_cols + j] = ptr[j * num_cols + i];
			ptr[j * num_cols + i] = tmp;
		}
	tmp = num_cols;
	num_cols = num_rows;
	num_rows = tmp;
	return *this;
}

template<class T> matrix<T>& matrix<T>::fill (T value) {
	T *ptr = data.data();
	size_t m = data.size();
	for (size_t i=0; i < m; ++i)
		ptr[i] = value;
	return *this;
}

template<class T> matrix<T>& matrix<T>::range (T v1, T v2) {
	T* ptr = data.data();
	size_t m = data.size();
	T step = (v2 - v1 + 1.0)/m;
	for (size_t i=0; i < m; ++i)
		ptr[i] = i*step + v1;
	return *this;
}

template<class T> matrix<T>& matrix<T>::randn () {
	srand(time(NULL));
	T *ptr = data.data();
	size_t m = data.size();
	for (size_t i=0; i < m; ++i)
		ptr[i] = rand() / float(RAND_MAX);
	return *this;
}

template<class T> matrix<T>& matrix<T>::add(T t) {
	for (size_t i=0; i < data.size(); ++i)
		data[i] += t;
	return *this;
}

template<class T> matrix<T>& matrix<T>::scale(T t) {
	for (size_t i=0; i < data.size(); ++i)
		data[i] *= t;
	return *this;
}

template<class T> matrix<T>& matrix<T>::power(T t) {
	for (size_t i=-0; i < data.size(); ++i)
		data[i] = pow(data[i], float(t));
	return *this;
}

template<class T> T matrix<T>::min () const {
	T min_value = numeric_limits<T>::max();
	const T *ptr = data.data();
	size_t m = data.size();
	for (size_t i=0; i < m; ++i) {
		if (data[i] < min_value)
			min_value = ptr[i];
	}
	return min_value;
}
template<class T> T matrix<T>::max () const {
	T max_value = numeric_limits<T>::min();
	const T *ptr = data.data();
	size_t m = data.size();
	for (size_t i=0; i < m; ++i) {
		if (data[i] > max_value)
			max_value = ptr[i];
	}
	return max_value;
}
template<class T> T matrix<T>::sum () const {
	T value = 0;
	const T *ptr = data.data();
	size_t m = data.size();
	for (size_t i=0; i < m; ++i)
		value += ptr[i];
	return value;
}
template<class T> matrix<T>& matrix<T>::max (int axis) const {
	assert (axis == 1 || axis == 0);
	if (axis == 0) {
		matrix<T> *vector = new matrix<T>(1, num_cols);
		T *ptr = data.data();
		for (size_t j=0; j < num_cols; ++j) {
			T max_value = numeric_limits<T>::min();
			for (size_t i=0; i < num_rows; ++i) {
				if (max_value < ptr[i*num_cols + j])
					max_value = ptr[i*num_cols + j];
			}
			(*vector)(0,j) = max_value;
		}
		return *vector;
	} else {
		matrix<T> *vector = new matrix<T>(num_rows, 1);
		T *ptr = data.data();
		for (size_t i=0; i < num_rows; ++i) {
			T max_value = numeric_limits<T>::min();
			for (size_t j=0; j < num_cols; ++j) {
				if (max_value < ptr[i*num_cols + j])
					max_value = ptr[i*num_cols + j];
			}
			(*vector)(i,0) = max_value;
		}
		return *vector;
	}
}
template<class T> matrix<T>& matrix<T>::min (int axis) const {
	assert (axis == 1 || axis == 0);
	if (axis == 0) {
		matrix<T> *vector = new matrix<T>(1, num_cols);
		T *ptr = data.data();
		for (size_t j=0; j < num_cols; ++j) {
			T min_value = numeric_limits<T>::max();
			for (size_t i=0; i < num_rows; ++i) {
				if (max_value > ptr[i*num_cols + j])
					max_value = ptr[i*num_cols + j];
			}
			(*vector)(0,j) = min_value;
		}
		return *vector;
	} else {
		matrix<T> *vector = new matrix<T>(num_rows, 1);
		T *ptr = data.data();
		for (size_t i=0; i < num_rows; ++i) {
			T min_value = numeric_limits<T>::max();
			for (size_t j=0; j < num_cols; ++j) {
				if (max_value > ptr[i*num_cols + j])
					max_value = ptr[i*num_cols + j];
			}
			(*vector)(i,0) = min_value;
		}
		return *vector;
	}
}
template<class T> matrix<T>& matrix<T>::sum (int axis) const {
	assert (axis == 1 || axis == 0);
	if (axis == 0) {
		matrix<T> *vector = new matrix<T>(1, num_cols);
		T *ptr = data.data();
		for (size_t j=0; j < num_cols; ++j) {
			T sum_value = 0.0;
			for (size_t i=0; i < num_rows; ++i)
				sum_value += ptr[i*num_cols + j];
			(*vector)(0,j) = sum_value;
		}
		return *vector;
	} else {
		matrix<T> *vector = new matrix<T>(num_rows, 1);
		T *ptr = data.data();
		for (size_t i=0; i < num_rows; ++i) {
			T sum_value = 0.0;
			for (size_t j=0; j < num_cols; ++j)
				sum_value += ptr[i*num_cols + j];
			(*vector)(i,0) = sum_value;
		}
		return *vector;
	}
}

template<class T> matrix<T>& matrix<T>::operator= (const matrix<T>& mx) {
	if (this != &mx) {
		if (this->num_rows != mx.rows() || this->num_cols != mx.cols()) {
			this->data.clear();
			this->num_cols = mx.cols();
			this->num_rows = mx.rows();
			this->data.resize(mx.size());
		}
		copy(mx.ptr(), mx.ptr() + mx.size(), data.data());
	}
	return *this;
}
template<class T> matrix<T>& matrix<T>::operator= (matrix<T>&& mx) {
	data.clear();
	num_cols = mx.cols();
	num_rows = mx.rows();
	
}

template<class T> matrix<T>& matrix<T>::operator= (const view<T>& mx) {
	if (this->num_rows != mx.rows() || this->num_cols != mx.cols()) {
		this->data.clear();
		this->num_cols = mx.cols();
		this->num_rows = mx.rows();
		this->data.resize(mx.size());
	}
	T *ptr = data.data();
	for (size_t i=0; i < num_rows; ++i)
		for (size_t j=0; j < num_cols; ++j)
			ptr[i*num_cols + j] = mx(i, j);
	return *this;
}

template<class T> matrix<T>& matrix<T>::add(const matrix<T>& mx) {
	assert(mx.rows() == this->rows() && mx.cols() == this->cols());
	const T *ptrs = mx.ptr();
	T *ptrd = data.data();
	size_t m = mx.size();
	for (size_t i = 0; i < m; ++i)
		ptrd[i] += ptrs[i];
	return *this;
}
template<class T> matrix<T>& matrix<T>::add(const view<T>& mx) {
	assert(mx.rows() == this->rows() && mx.cols() == this->cols());
	T *ptrd = data.data();
	for (size_t i=0; i < num_rows; ++i)
		for (size_t j=0; j < num_cols; ++j)
			ptrd[i*num_cols + j] = mx(i,j);
	return *this;
}
template<class T> matrix<T>& matrix<T>::mul(const matrix<T>& mx) {
	assert(mx.rows() == this->rows() && mx.cols() == this->cols());
	const T *ptrs = mx.ptr();
	T *ptrd = data.data();
	size_t m = mx.size();
	for (size_t i = 0; i < m; ++i)
		ptrd[i] *= ptrs[i];
	return *this;
}
template<class T> matrix<T>& matrix<T>::mul(const matrix<T>& mx) {
	assert(mx.rows() == this->rows() && mx.cols() == this->cols());
	T *ptrd = data.data();
	for (size_t i=0; i < num_rows; ++i)
		for (size_t j=0; j < num_cols; ++j)
			ptrd[i*num_cols + j] *= mx(i,j);
	return *this;
}
template<class T> matrix<T>& matrix<T>::dot(const matrix<T>& mx) {
	assert(this->num_cols == mx.rows());
	size_t p = mx.cols();
	matrix<T> tmp(num_rows, mx.cols());
	tmp.fill(0);
	const T *ptr_a = mx.ptr();
	T *ptr_b = data.data();
	T *ptr_c = tmp.mutable_ptr();
	#pragma omp parallel for
	for (size_t i=0; i < num_rows; ++i)
		for (size_t j=0; j < p; ++j)
			for (size_t k=0; k < num_cols; ++k)
				ptr_c[i * p + j] += ptr_b[i * num_cols + k] * ptr_a[j * p + k];
	*this = tmp;
	return *this;
}
template<class T> matrix<T>& matrix<T>::dot(const view<T>& mx) {
	assert(this->num_cols == mx.rows());
	size_t p = mx.cols();
	matrix<T> tmp(num_rows, mx.cols());
	tmp.fill(0);
	const T *ptr_a = mx.ptr();
	T *ptr_b = data.data();
	T *ptr_c = tmp.mutable_ptr();
	#pragma omp parallel for
	for (size_t i=0; i < num_rows; ++i)
		for (size_t j=0; j < p; ++j)
			for (size_t k=0; k < num_cols; ++k)
				ptr_c[i * p + j] += ptr_b[i * num_cols + k] * mx(j,k);
	*this = tmp;
	return *this;
}

template<class T> size_t matrix<T>::rows() const { return num_rows; }
template<class T> size_t matrix<T>::cols() const { return num_cols; }
template<class T> size_t matrix<T>::size() const { return num_rows * num_cols; }

template<class T> void matrix<T>::size(size_t rows, size_t cols) {
	num_rows = rows;
	num_cols = cols;
	data.clear();
	data.resize(num_rows * num_cols);
}

template<class T> const T* matrix<T>::ptr() const { return data.data(); }
template<class T> T* matrix<T>::mutable_ptr() { return data.data(); }

template<class T> matrix<T>::matrix (const matrix<T>& mx) {
	this->num_rows = mx.num_rows;
	this->num_cols = mx.num_cols;
	this->data = mx.data;
}

template<class T> matrix<T>::matrix (matrix<T>&& mx) {
	data.clear();
	num_rows = mx.num_rows;
	num_cols = mx.num_cols;
	data = std::move(mx.data);
}

template<class T> matrix<T>::~matrix() {
	data.clear();
}

template<class T> matrix<T>::matrix ()
	: num_rows(0),
	  num_cols(0),
	  data(0) {}

template<class T> matrix<T>::matrix (size_t rows, size_t cols)
	: num_rows(rows),
	  num_cols(cols),
	  data(rows * cols) {}

template<class T> matrix<T>::matrix (T* raw_data, size_t rows, size_t cols)
	: num_rows(rows),
	  num_cols(cols),
	  data(raw_data, raw_data + rows*cols) {}

template<class T> T& matrix<T>::operator () (size_t i, size_t j)
{
	return data[i * num_cols + j];
}

template<class T> T matrix<T>::operator () (size_t i, size_t j) const
{
	return data[i * num_cols + j];
}

template<class T> matrix<T>* matrix<T>::load (const char *filename, size_t rows, size_t cols) {
	ifstream file (filename, ios::binary | ios::in | ios::ate);
	if (file.is_open()) {
		streampos size = file.tellg();
		size_t array_len = size / sizeof(T);
		if (array_len != rows * cols) {
			cout << "array dimension mismatch!";
			return 0;
		}
		file.seekg(0, ios::beg);
		T *memblock = new T[array_len];
		file.read((char*)memblock, size);
		file.close();
		matrix<T>* data = new matrix<T>(memblock, rows, cols);
		delete memblock;
		return data;
	}
	else {
		cout << "Input file not found";
		return 0;
	}
}

template<class T>
void matrix<T>::dump (const char* filename, matrix<T>& mx) {
	ofstream file(filename, ios::binary | ios::out);
	if (file.is_open()){
		file.write((char*)(mx.mutable_ptr()), mx.size()*sizeof(T));
		file.close();
	}
	else {
		cout << "Dumping failed";
		return;
	}
}

//the following are UBUNTU/LINUX ONLY terminal color codes.
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */
template<class T>
ostream& operator<< (ostream& os, const matrix<T>& mx) {
	os << BOLDGREEN << "[ " << RESET;
	for (int i=0; i < mx.rows(); ++i) {
		for (int j=0; j < mx.cols(); ++j) {
			os << YELLOW << mx(i,j) << RESET;
			if (j < mx.cols() - 1)
				os << ",\t";
		}
		if (i < mx.rows() - 1)
			os << "\n  ";
	}
	os << BOLDGREEN << " ]" << RESET;
	return os;
}

template<class T>
matrix<bool>&  operator== (const matrix<T>& mx, const T value) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] == value)
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}

template<class T>
matrix<bool>&  operator!= (const matrix<T>& mx, const T value) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] != value)
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}

template<class T>
matrix<bool>&  operator> (const matrix<T>& mx, const T value) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] > value)
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}
template<class T>
matrix<bool>&  operator>= (const matrix<T>& mx, const T value) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] >= value)
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}
template<class T>
matrix<bool>&  operator< (const matrix<T>& mx, const T value) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] < value)
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}
template<class T>
matrix<bool>&  operator<= (const matrix<T>& mx, const T value) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] <= value)
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}


template<class T> matrix<bool>&  operator== (const T value, const matrix<T>& mx) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] == value)
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}
template<class T> matrix<bool>&  operator!= (const T value, const matrix<T>& mx) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] != value)
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}
template<class T> matrix<bool>&  operator> (const T value, const matrix<T>& mx) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] > value)
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}
template<class T> matrix<bool>&  operator>= (const T value, const matrix<T>& mx) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] >= value)
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}
template<class T> matrix<bool>&  operator< (const T value, const matrix<T>& mx) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] < value)
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}
template<class T> matrix<bool>&  operator<= (const T value, const matrix<T>& mx) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] <= value)
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}
//
template<class T>
matrix<bool>&  operator== (const matrix<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	const T *ptr2 = mx2.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] == ptr2[i])
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}

template<class T>
matrix<bool>&  operator!= (const matrix<T>& mx, const matrix<.& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	const T *ptr2 = mx2.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] != ptr2[i])
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}

template<class T>
matrix<bool>&  operator> (const matrix<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	const T *ptr2 = mx2.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] > ptr2[i])
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}
template<class T>
matrix<bool>&  operator>= (const matrix<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	const T *ptr2 = mx2.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] >= ptr2[i])
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}
template<class T>
matrix<bool>&  operator< (const matrix<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	const T *ptr2 = mx2.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] < ptr2[i])
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}
template<class T>
matrix<bool>&  operator<= (const matrix<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	const T *ptr2 = mx2.ptr();
	T *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] <= ptr2[i])
			ptr_i[i] = true;
		else
			ptr_i[i] = false;
	}
	return *index;
}

template<class T> matrix<T>& operator+ (const matrix<T>& A, const matrix<T>& B) {
	assert (A.cols() == B.cols() && A.rows() == B.rows());
	const T *ptr_a, *ptr_b;
	ptr_a = A.ptr();
	ptr_b = B.ptr();
  matrix<T> *C = new matrix<T>(A.rows(), A.cols());
	T *ptr_c = C->mutable_ptr();
	size_t m = A.size();
	for (size_t i=0; i < m; ++i)
		ptr_c[i] = ptr_a[i] + ptr_b[i];
  return *C;
}
template<class T> matrix<T>& operator- (const matrix<T>& A, const matrix<T>& B) {
	assert (A.cols() == B.cols() && A.rows() == B.rows());
	const T *ptr_a, *ptr_b;
	ptr_a = A.ptr();
	ptr_b = B.ptr();
  matrix<T> *C = new matrix<T>(A.rows(), A.cols());
	T *ptr_c = C->mutable_ptr();
	size_t m = A.size();
	for (size_t i=0; i < m; ++i)
		ptr_c[i] = ptr_a[i] - ptr_b[i];
  return *C;
}
template<class T> matrix<T>& operator* (const matrix<T>& A, const matrix<T>& B) {
	assert (A.cols() == B.cols() && A.rows() == B.rows());
	const T *ptr_a, *ptr_b;
	ptr_a = A.ptr();
	ptr_b = B.ptr();
  matrix<T> *C = new matrix<T>(A.rows(), A.cols());
	T *ptr_c = C->mutable_ptr();
	size_t m = A.size();
	for (size_t i=0; i < m; ++i)
		ptr_c[i] = ptr_a[i] * ptr_b[i];
  return *C;
}
template<class T> matrix<T>& operator/ (const matrix<T>& A, const matrix<T>& B) {
	assert (A.cols() == B.cols() && A.rows() == B.rows());
	const T *ptr_a, *ptr_b;
	ptr_a = A.ptr();
	ptr_b = B.ptr();
  matrix<T> *C = new matrix<T>(A.rows(), A.cols());
	T *ptr_c = C->mutable_ptr();
	size_t m = A.size();
	for (size_t i=0; i < m; ++i)
		ptr_c[i] = ptr_a[i] / ptr_b[i];
  return *C;
}
template<class T> matrix<T>& operator^ (const matrix<T>& A, T t) {
	assert(A.size() > 0);
	T *ptr = A.mutable_ptr();
	size_t m = A.size();
	for (size_t i=0; i < m; ++i)
		ptr[i] = pow(ptr[i], t);
}

template<class T> matrix<T>& operator+ (const matrix<T>& A, const T t) {
	const T *ptr_a;
	ptr_a = A.ptr();
	matrix<T> *C = new matrix<T>(A.rows(), A.cols());
	T *ptr_c = C->mutable_ptr();
	size_t m = A.size();
	for (size_t i=0; i < m; ++i)
		ptr_c[i] = ptr_a[i] + t;
  return *C;
}
template<class T> matrix<T>& operator- (const matrix<T>& A, const T t) {
	const T *ptr_a;
	ptr_a = A.ptr();
	matrix<T> *C = new matrix<T>(A.rows(), A.cols());
	T *ptr_c = C->mutable_ptr();
	size_t m = A.size();
	for (size_t i=0; i < m; ++i)
		ptr_c[i] = ptr_a[i] - t;
  return *C;
}
template<class T> matrix<T>& operator* (const matrix<T>& A, const T t) {
	const T *ptr_a;
	ptr_a = A.ptr();
	matrix<T> *C = new matrix<T>(A.rows(), A.cols());
	T *ptr_c = C->mutable_ptr();
	size_t m = A.size();
	for (size_t i=0; i < m; ++i)
		ptr_c[i] = ptr_a[i] * t;
  return *C;
}
template<class T> matrix<T>& operator/ (const matrix<T>& A, const T t) {
	const T *ptr_a;
	ptr_a = A.ptr();
	matrix<T> *C = new matrix<T>(A.rows(), A.cols());
	T *ptr_c = C->mutable_ptr();
	size_t m = A.size();
	for (size_t i=0; i < m; ++i)
		ptr_c[i] = ptr_a[i] / t;
  return *C;
}

template<class T> matrix<T>& operator+ (const T t, const matrix<T>& A) {
	const T *ptr_a;
	ptr_a = A.ptr();
	matrix<T> *C = new matrix<T>(A.rows(), A.cols());
	T *ptr_c = C->mutable_ptr();
	size_t m = A.size();
	for (size_t i=0; i < m; ++i)
		ptr_c[i] = ptr_a[i] + t;
  return *C;
}
template<class T> matrix<T>& operator- (const T t, const matrix<T>& A) {
	const T *ptr_a;
	ptr_a = A.ptr();
	matrix<T> *C = new matrix<T>(A.rows(), A.cols());
	T *ptr_c = C->mutable_ptr();
	size_t m = A.size();
	for (size_t i=0; i < m; ++i)
		ptr_c[i] = t - ptr_a[i];
  return *C;
}
template<class T> matrix<T>& operator* (const T t, const matrix<T>& A) {
	const T *ptr_a;
	ptr_a = A.ptr();
	matrix<T> *C = new matrix<T>(A.rows(), A.cols());
	T *ptr_c = C->mutable_ptr();
	size_t m = A.size();
	for (size_t i=0; i < m; ++i)
		ptr_c[i] = ptr_a[i] * t;
  return *C;
}
template<class T> matrix<T>& operator/ (const T t, const matrix<T>& A) {
	const T *ptr_a;
	ptr_a = A.ptr();
	matrix<T> *C = new matrix<T>(A.rows(), A.cols());
	T *ptr_c = C->mutable_ptr();
	size_t m = A.size();
	for (size_t i=0; i < m; ++i)
		ptr_c[i] = t / ptr_a[i];
  return *C;
}

// slicing methods
template<class T> view<T>& matrix<T>::r_ (const vector<size_t>& row_indices) {
	vector<size_t> cols, rows;
	for (size_t i=0; i < num_cols; ++i)
		cols.push_back(i);
	for (size_t j=0; j < row_indices.size(); ++j) {
		if (row_indices[j] < num_rows && row_indices[j] >= 0)
			rows.push_back(row_indices[j]);
		else
			assert (1==0);
	}
	view<T> *v = new view<T>(rows.size(), cols().size(), cols, rows, data.data());
	return *v;
}
template<class T> view<T>& matrix<T>::r_ (const matrix<bool>& rows) {
	assert (rows.cols() == 1 && rows.rows() == num_rows);
	vector<size_t> cols, rows;
	for (size_t i=0; i < num_cols; ++i)
		cols.push_back(i);
	for (size_t j=0; j < rows.size(); ++j)
		if (rows(j,0) == true)
			rows.push_back(j);
	view<T> *v = new view<T>(rows.size(), cols().size(), cols, rows, data.data());
	return *v;
}
template<class T> view<T>& matrix<T>::r_ (size_t r1, size_t r2) {
	assert ((r1 >= 0 && r1 <= r2 && r2 < num_rows) ||
					(r1 >= 0 && r1 < num_rows && r2 == matrix<T>::END) ||
					(r1 == r2 == matrix<T>::END));
	vector<size_t> cols, rows;
	for (size_t i=0; i < num_cols; ++i)
		cols.push_back(i);
	if (r2 == matrix<T>::END)
		r2 = num_rows - 1;
		if (r1 == matrix<T>::END)
			r1 = num_rows - 1;
	for (size_t j=r1; j <= r2; ++j)
		rows.push_back(j);
	view<T> *v = new view<T>(rows.size(), cols().size(), cols, rows, data.data());
	return *v;
}

template<class T> view<T>& matrix<T>::c_ (const vector<size_t>& cols) {
	vector<size_t> cols, rows;
	for (size_t i=0; i < num_rows; ++i)
		rows.push_back(i);
	for (size_t j=0; j < cols.size(); ++j) {
		if (cols[j] < num_cols && cols[j] >= 0)
			cols.push_back(cols[j]);
		else
			assert (1==0);
	}
	view<T> *v = new view<T>(rows.size(), cols().size(), cols, rows, data.data());
	return *v;
}
template<class T> view<T>& matrix<T>::c_ (const matrix<bool>& cols) {
	assert (cols.rows() == 1 && cols.cols() == num_cols);
	vector<size_t> cols, rows;
	for (size_t i=0; i < num_rows; ++i)
		rows.push_back(i);
	for (size_t j=0; j < cols.size(); ++j)
		if (cols(0,j) == true)
			cols.push_back(j);
	view<T> *v = new view<T>(rows.size(), cols().size(), cols, rows, data.data());
	return *v;
}
template<class T> view<T>& matrix<T>::c_ (size_t c1, size_t c2) {
	assert ((c1 >= 0 && c1 <= c2 && c2 < num_rows) ||
					(c1 >= 0 && c1 < num_rows && c2 == matrix<T>::END) ||
					(c1 == c2 == matrix<T>::END));
	vector<size_t> cols, rows;
	for (size_t i=0; i < num_rows; ++i)
		rows.push_back(i);
	if (c2 == matrix<T>::END)
		c2 = num_cols - 1;
		if (c1 == matrix<T>::END)
			c1 = num_cols - 1;
	for (size_t j=c1; j <= c2; ++j)
		cols.push_back(j);
	view<T> *v = new view<T>(rows.size(), cols().size(), cols, rows, data.data());
	return *v;
}

template<class T> view<T>& matrix<T>::grid (const matrix<bool>& rows, const matrix<bool>& cols) {
	assert (rows.rows() == num_rows && rows.cols() == 1 && cols.rows() == 1 && cols.cols() == num_cols);
	vector<size_t> cols, rows;
	for (size_t i=0; i < num_rows; ++i)
		rows.push_back(i);
	for (size_t i=0; i < rows.size(); ++i)
		if (rows(0,i) == true)
			rows.push_back(i);
	for (size_t j=0; j < cols.size(); ++j)
		if (logic_indices(j,0) == true)
			cols.push_back(j);
	view<T> *v = new view<T>(rows.size(), cols().size(), cols, rows, data.data());
	return *v;
}

template<class T> view<T>& matrix<T>::grid (const vector<size_t>& rows, const std::vector<size_t>& cols) {
	assert(rows.size() > 0 && cols.size() > 0);
	vector<size_t> cols, rows;
	for (size_t i=0; i < rows.size(); ++i) {
		if (rows[i] >= 0 && rows[i] < num_rows)
			rows.push_back(rows[i]);
		else
			assert (1==0);
	}
	for (size_t i=0; i < cols.size(); ++i) {
		if (cols[i] >= 0 && cols[i] < num_cols)
			cols.push_back(cols[i]);
		else
			assert (1==0);
	}
	view<T> *v = new view<T>(rows.size(), cols().size(), cols, rows, data.data());
	return *v;
}
template<class T> view<T>& matrix<T>::area (size_t r1, size_t r2, size_t c1, size_t c2) {
	assert ((r1 >= 0 && r1 <= r2 && r2 < num_rows) ||
					(r1 >= 0 && r1 < num_rows && r2 == matrix<T>::END) ||
					(r1 == r2 && r2 == matrix<T>::END) ||
					(c1 >= 0 && c1 <= c2 && c2 < num_cols) ||
					(c1 >= 0 && c1 < num_cols && c2 == matrix<T>::END) ||
					(c1 == c2 && c2 == matrix<T>::END) );
					vector<size_t> cols, rows;
	if (r2 == matrix<T>::END)
		r2 = num_rows - 1;
	if (r1 == matrix<T>::END)
		f1 = num_rows - 1;
	for (size_t i=r1; i <= r2; ++i)
		rows.push_back(i);
	if (c2 == matrix<T>::END)
		c2 = num_cols - 1;
		if (c1 == matrix<T>::END)
			c1 = num_cols - 1;
	for (size_t j=c1; j <= c2; ++j)
		cols.push_back(j);
	view<T> *v = new view<T>(rows.size(), cols().size(), cols, rows, data.data());
	return *v;
}
#endif
