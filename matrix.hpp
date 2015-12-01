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
#include <iterator>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "constants.hpp"

using namespace std;
template<class T> class view;
template<class T>
class matrix
{
public:
		matrix ();
		matrix (size_t rows, size_t cols);
		matrix (T *raw_data, size_t rows, size_t cols);
		matrix (const matrix<T>& mx);
		matrix (const view<T>& mx);
		// matrix (matrix<T>&& mx);
		~matrix ();

		const T* ptr() const;
		T* mutable_ptr();

		// slicing, indexing operators
		T& operator() (size_t i, size_t j);
		T operator() (size_t i, size_t j) const;

		size_t rows() const;
		size_t cols() const;
		size_t size() const;
		matrix<T>& size(size_t rows, size_t cols);
		matrix<size_t>& shape() const;
		size_t shape(int axis) const;

		// arimethic operators
		matrix<T>& add(T t);
		matrix<T>& scale(T t);
		matrix<T>& power(T t);

		matrix<T>& t ();
		matrix<T>& reshape (size_t a1, size_t a2);

		T max () const;
		T min () const;
		T sum () const;
		T mean() const;
		matrix<T>& max (int axis) const;
		matrix<T>& min (int axis) const;
		matrix<T>& sum (int axis) const;
		matrix<T>& mean (int axis) const;

		// fillers
		matrix<T>& fill (T value);
		matrix<T>& range (T v1, T v2);
		matrix<T>& randn ();

		// slicing methods
		view<T>& r_ (const vector<size_t>& rows);
		view<T>& r_ (const matrix<unsigned char>& rows);
		view<T>& r_ (int r1, int r2);
		view<T>& r_ (int r);

		view<T>& c_ (const vector<size_t>& cols);
		view<T>& c_ (const matrix<unsigned char>& cols);
		view<T>& c_ (int c1, int c2);
		view<T>& c_ (int c);

		view<T>& grid (const matrix<unsigned char>& rows, const matrix<unsigned char>& cols);
		view<T>& grid (const vector<size_t>& rows, const std::vector<size_t>& cols);
		view<T>& area (int r1, int r2, int c1, int c2);

		// assignment operators
		// matrix<T>& operator= (matrix<T>&& mx);
		matrix<T>& operator= (const matrix<T>& mx);
		matrix<T>& operator= (const view<T>& mx);
		// matrix<T>& operator= (const T t);

		// element-wise operators
		matrix<T>& add(const matrix<T>& mx);
		matrix<T>& mul(const matrix<T>& mx);
		matrix<T>& add(const view<T>& mx);
		matrix<T>& mul(const view<T>& mx);

		// dot-product operator
		matrix<T>& dot(const matrix<T>& mx);
		matrix<T>& dot(const view<T>& mx);

		// static methods
		static matrix<T>& load (const char* filename, size_t rows, size_t cols);
		static void 			dump (const char* filename, matrix<T>& mx);
		static matrix<T>& tile (const matrix<T>& pattern, size_t nrows, size_t ncols);

		// some constants
		static int const END;
		static int const ALL;
		static int const DIM_VERTICAL;
		static int const DIM_HORIZONTAL;
		static int const DIM_0;
		static int const DIM_1;
		static int const DIM_FLAT;
		static int const DIM_FIRST;
		static int const DIM_LAST;
		static int const SORT_ASCEND;
		static int const SORT_DESCEND;
		static int const SORT_ROWS;
		static int const SORT_COLS;
		static int const SORT_FLAT;

private:
		size_t num_rows;
		size_t num_cols;
		vector<T> data;
};

template<class T> int const matrix<T>::END = -1;
template<class T> int const matrix<T>::ALL = -2;
template<class T> int const matrix<T>::DIM_VERTICAL = 0;
template<class T> int const matrix<T>::DIM_HORIZONTAL = 1;
template<class T> int const matrix<T>::DIM_0 = 0;
template<class T> int const matrix<T>::DIM_1 = 1;
template<class T> int const matrix<T>::DIM_FLAT = -1;
template<class T> int const matrix<T>::DIM_FIRST = 0;
template<class T> int const matrix<T>::DIM_LAST = 1;
template<class T> int const matrix<T>::SORT_ASCEND = -3;
template<class T> int const matrix<T>::SORT_DESCEND = -4;
template<class T> int const matrix<T>::SORT_ROWS = 1;
template<class T> int const matrix<T>::SORT_COLS = 0;
template<class T> int const matrix<T>::SORT_FLAT = -1;

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

template<class T> matrix<unsigned char>&  operator< (const matrix<T>& mx, const matrix<T>& mx2);
template<class T> matrix<unsigned char>&  operator<= (const matrix<T>& mx, const matrix<T>& mx2);
template<class T> matrix<unsigned char>&  operator> (const matrix<T>& mx, const matrix<T>& mx2);
template<class T> matrix<unsigned char>&  operator>= (const matrix<T>& mx, const matrix<T>& mx2);
template<class T> matrix<unsigned char>&  operator== (const matrix<T>& mx, const matrix<T>& mx2);
template<class T> matrix<unsigned char>&  operator!= (const matrix<T>& mx, const matrix<T>& mx2);

template<class T> matrix<unsigned char>&  operator< (const matrix<T>& mx, const T t);
template<class T> matrix<unsigned char>&  operator<= (const matrix<T>& mx, const T t);
template<class T> matrix<unsigned char>&  operator> (const matrix<T>& mx, const T t);
template<class T> matrix<unsigned char>&  operator>= (const matrix<T>& mx, const T t);
template<class T> matrix<unsigned char>&  operator== (const matrix<T>& mx, const T t);
template<class T> matrix<unsigned char>&  operator!= (const matrix<T>& mx, const T t);

template<class T> matrix<unsigned char>&  operator< (const T t, const matrix<T>& mx);
template<class T> matrix<unsigned char>&  operator<= (const T t, const matrix<T>& mx);
template<class T> matrix<unsigned char>&  operator> (const T t, const matrix<T>& mx);
template<class T> matrix<unsigned char>&  operator>= (const T t, const matrix<T>& mx);
template<class T> matrix<unsigned char>&  operator== (const T t, const matrix<T>& mx);
template<class T> matrix<unsigned char>&  operator!= (const T t, const matrix<T>& mx);

template<class T> matrix<T>& operator^ (const matrix<T>& A, const T t);

bool _sort_descending (float i, float j);
bool des_comparator (const pair<float, size_t>& l, const pair<float, size_t>& r);
bool asc_comparator (const pair<float, size_t>& l, const pair<float, size_t>& r);
matrix<float>& matrix_sort (const matrix<float>& mx, int axis=-1, bool ascending=true);
matrix<float>& matrix_sort (const view<float>& mx, int axis=-1, bool ascending=true);
matrix<size_t>& argsort (const matrix<float>& mx, int axis=-1, bool ascending=true);
matrix<size_t>& argsort (const view<float>& mx, int axis=-1, bool ascending=true);

matrix<double>& blas_matrix_dot (const matrix<double>& A, const matrix<double>& B);
matrix<float>& blas_matrix_dot (const matrix<float>& A, const matrix<float>& B);
matrix<float>& blas_pairwise_distance (const matrix<float>& A, const matrix<float>& B);
matrix<double>& blas_pairwise_distance (const matrix<double>& A, const matrix<double>& B);

void 					cuda_init (int& dev_id);
matrix<float>& cuda_matrix_dot (const matrix<float>& A, const matrix<float>& B);
matrix<float>& cuda_pairwise_distance (const matrix<float>& A, const matrix<float>& B);
matrix<float>& cuda_pairwise_distance (const matrix<float>& A);
////////////////////////////////////////////////////////////////////////////////
template<class T> matrix<T>::matrix (const matrix<T>& mx) {
	// cout << endl << "*copy constructor*" << endl;
	this->num_rows = mx.num_rows;
	this->num_cols = mx.num_cols;
	this->data = mx.data;
}

template<class T> matrix<T>::matrix (const view<T>& mx) {
	// cout << endl << "*copy constructor*" << endl;
	this->num_rows = mx.steps.size();
	this->num_cols = mx.strides.size();
	size_t m = num_rows * num_cols;
	data.resize(m);
	for (size_t j=0; j < num_rows; ++j)
		for (size_t i=0; i < num_cols; ++i)
			data.push_back(mx(j,i));
}
template<class T> matrix<T>& matrix<T>::operator= (const matrix<T>& mx) {
	if (this != &mx) {
		// cout << endl << "*copy assignment*" << endl;
		if (this->num_rows != mx.rows() || this->num_cols != mx.cols()) {
			this->data.clear();
			this->num_cols = mx.cols();
			this->num_rows = mx.rows();
			// this->data.resize(mx.size());
		}
		data = mx.data;
		// copy(mx.ptr(), mx.ptr() + mx.size(), data.data());
	}
	return *this;
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
template<class T> matrix<T>& matrix<T>::t () {
	int mn1 = size() - 1;
	std::vector<bool> visited(size());
	int ix = 0;
	T *first = data.data();
	T *last = data.data() + size();
	T *cycle = first;
	while (++cycle != last) {
		++ix;
		if (visited[ix])
			continue;
		int a = ix;
		do {
			a = a == mn1 ? mn1 : (num_rows * a) % mn1;
			std::swap (*(first + a), *cycle);
			visited[a] = true;
		} while ((first + a) != cycle);
	}
	size_t tmp = num_cols;
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
template<class T> T matrix<T>::mean () const {
	T value = 0;
	const T *ptr = data.data();
	size_t m = data.size();
	for (size_t i=0; i < m; ++i)
		value += ptr[i];
	return value/data.size();
}
template<class T> matrix<T>& matrix<T>::max (int axis) const {
	assert (axis == 1 || axis == 0);
	if (axis == 0) {
		matrix<T> *vector = new matrix<T>(1, num_cols);
		const T *ptr = data.data();
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
		const T *ptr = data.data();
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
		const T *ptr = data.data();
		for (size_t j=0; j < num_cols; ++j) {
			T min_value = numeric_limits<T>::max();
			for (size_t i=0; i < num_rows; ++i) {
				if (min_value > ptr[i*num_cols + j])
					min_value = ptr[i*num_cols + j];
			}
			(*vector)(0,j) = min_value;
		}
		return *vector;
	} else {
		matrix<T> *vector = new matrix<T>(num_rows, 1);
		const T *ptr = data.data();
		for (size_t i=0; i < num_rows; ++i) {
			T min_value = numeric_limits<T>::max();
			for (size_t j=0; j < num_cols; ++j) {
				if (min_value > ptr[i*num_cols + j])
					min_value = ptr[i*num_cols + j];
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
		const T *ptr = data.data();
		for (size_t j=0; j < num_cols; ++j) {
			T sum_value = 0.0;
			for (size_t i=0; i < num_rows; ++i)
				sum_value += ptr[i*num_cols + j];
			(*vector)(0,j) = sum_value;
		}
		return *vector;
	} else {
		matrix<T> *vector = new matrix<T>(num_rows, 1);
		const T *ptr = data.data();
		for (size_t i=0; i < num_rows; ++i) {
			T sum_value = 0.0;
			for (size_t j=0; j < num_cols; ++j)
				sum_value += ptr[i*num_cols + j];
			(*vector)(i,0) = sum_value;
		}
		return *vector;
	}
}
template<class T> matrix<T>& matrix<T>::mean (int axis) const {
	assert (axis == 1 || axis == 0);
	if (axis == 0) {
		matrix<T> *vector = new matrix<T>(1, num_cols);
		const T *ptr = data.data();
		for (size_t j=0; j < num_cols; ++j) {
			T sum_value = 0.0;
			for (size_t i=0; i < num_rows; ++i)
				sum_value += ptr[i*num_cols + j];
			(*vector)(0,j) = sum_value/num_rows;
		}
		return *vector;
	} else {
		matrix<T> *vector = new matrix<T>(num_rows, 1);
		const T *ptr = data.data();
		for (size_t i=0; i < num_rows; ++i) {
			T sum_value = 0.0;
			for (size_t j=0; j < num_cols; ++j)
				sum_value += ptr[i*num_cols + j];
			(*vector)(i,0) = sum_value/num_cols;
		}
		return *vector;
	}
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
template<class T> matrix<T>& matrix<T>::mul(const view<T>& mx) {
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

template<class T> matrix<T>& matrix<T>::size(size_t rows, size_t cols) {
	num_rows = rows;
	num_cols = cols;
	data.clear();
	data.resize(num_rows * num_cols);
	return *this;
}

template<class T> matrix<size_t>& matrix<T>::shape() const {
	matrix<size_t>* mx = new matrix<size_t>(1,2);
	(*mx)(0,0) = num_rows;
	(*mx)(0,1) = num_cols;
	return *mx;
}
template<class T> size_t matrix<T>::shape(int axis) const{
	assert(axis == 0 || axis == 1);
	if (axis == 0)
		return num_rows;
	else
		return num_cols;
}
template<class T> matrix<T>& matrix<T>::reshape (size_t a1, size_t a2) {
	assert (num_rows * num_cols == a1 * a2 && a1 > 0 && a2 > 0);
	num_rows = a1;
	num_cols = a2;
	return *this;
}

template<class T> const T* matrix<T>::ptr() const { return data.data(); }
template<class T> T* matrix<T>::mutable_ptr() { return data.data(); }

template<class T> matrix<T>::~matrix() {
	cout << endl << "*destructed*" << endl;
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

template<class T> matrix<T>& matrix<T>::load (const char *filename, size_t rows, size_t cols) {
	ifstream file (filename, ios::binary | ios::in | ios::ate);
	assert (file.is_open());
	streampos size = file.tellg();
	size_t array_len = size / sizeof(T);
	assert (array_len == rows * cols);
	file.seekg(0, ios::beg);
	T *memblock = new T[array_len];
	file.read((char*)memblock, size);
	file.close();
	matrix<T>* data = new matrix<T>(memblock, rows, cols);
	delete memblock;
	return *data;
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

template<class T>
ostream& operator<< (ostream& os, const matrix<T>& mx) {
	os << BOLDGREEN << "[ " << RESET;
	for (size_t i=0; i < mx.rows(); ++i) {
		for (size_t j=0; j < mx.cols(); ++j) {
			os << YELLOW << 1.f*mx(i,j) << RESET;
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
matrix<unsigned char>&  operator== (const matrix<T>& mx, const T value) {
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] == value)
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}

template<class T>
matrix<unsigned char>&  operator!= (const matrix<T>& mx, const T value) {
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] != value)
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}

template<class T>
matrix<unsigned char>&  operator> (const matrix<T>& mx, const T value) {
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] > value)
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}
template<class T>
matrix<unsigned char>&  operator>= (const matrix<T>& mx, const T value) {
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] >= value)
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}
template<class T>
matrix<unsigned char>&  operator< (const matrix<T>& mx, const T value) {
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] < value)
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}
template<class T>
matrix<unsigned char>&  operator<= (const matrix<T>& mx, const T value) {
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] <= value)
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}


template<class T> matrix<T>&  operator== (const T value, const matrix<T>& mx) {
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] == value)
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}
template<class T> matrix<unsigned char>&  operator!= (const T value, const matrix<T>& mx) {
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] != value)
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}
template<class T> matrix<unsigned char>&  operator> (const T value, const matrix<T>& mx) {
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] > value)
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}
template<class T> matrix<unsigned char>&  operator>= (const T value, const matrix<T>& mx) {
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] >= value)
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}
template<class T> matrix<unsigned char>&  operator< (const T value, const matrix<T>& mx) {
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] < value)
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}
template<class T> matrix<unsigned char>&  operator<= (const T value, const matrix<T>& mx) {
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] <= value)
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}
//
template<class T>
matrix<unsigned char>&  operator== (const matrix<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	const T *ptr2 = mx2.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] == ptr2[i])
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}

template<class T>
matrix<unsigned char>&  operator!= (const matrix<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	const T *ptr2 = mx2.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] != ptr2[i])
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}

template<class T>
matrix<unsigned char>&  operator> (const matrix<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	const T *ptr2 = mx2.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] > ptr2[i])
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}
template<class T>
matrix<unsigned char>&  operator>= (const matrix<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	const T *ptr2 = mx2.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] >= ptr2[i])
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}
template<class T>
matrix<unsigned char>&  operator< (const matrix<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	const T *ptr2 = mx2.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] < ptr2[i])
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
	}
	return *index;
}
template<class T>
matrix<unsigned char>&  operator<= (const matrix<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<unsigned char> *index = new matrix<unsigned char>(mx.rows(), mx.cols());
	const T *ptr = mx.ptr();
	const T *ptr2 = mx2.ptr();
	unsigned char *ptr_i = index->mutable_ptr();
	size_t m = mx.size();
	for (size_t i=0; i < m; ++i) {
		if (ptr[i] <= ptr2[i])
			ptr_i[i] = 1;
		else
			ptr_i[i] = 0;
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
	matrix<T> *C = new matrix<T>(A.rows(), A.cols());
	T *ptr = C->mutable_ptr();
	const T *ptr_ = A.ptr();
	size_t m = A.size();
	for (size_t i=0; i < m; ++i)
		ptr[i] = pow(ptr_[i], t);
	return *C;
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
		if (row_indices[j] < num_rows)
			rows.push_back(row_indices[j]);
		else
			assert (1==0);
	}
	view<T> *v = new view<T>(num_rows, num_cols, cols, rows, data.data());
	return *v;
}
template<class T> view<T>& matrix<T>::r_ (const matrix<unsigned char>& rows_) {
	assert ((rows_.cols() == 1 && rows_.rows() == num_rows) ||
					(rows_.rows() == 1 && rows_.cols() == num_rows));
	vector<size_t> cols, rows;
	for (size_t i=0; i < num_cols; ++i)
		cols.push_back(i);
	for (size_t j=0; j < rows_.size(); ++j)
		if (rows_(j,0) == 1)
			rows.push_back(j);
	view<T> *v = new view<T>(num_rows, num_cols, cols, rows, data.data());
	return *v;
}
template<class T> view<T>& matrix<T>::r_ (int r1, int r2) {
	assert ((r1 <= r2 && r2 < num_rows) ||
					(r1 < num_rows && r2 == matrix<T>::END) ||
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
	view<T> *v = new view<T>(num_rows, num_cols, cols, rows, data.data());
	return *v;
}
template<class T> view<T>& matrix<T>::r_ (int r) {
	assert (r < num_rows);
	vector<size_t> cols, rows;
	for (size_t i=0; i < num_cols; ++i)
		cols.push_back(i);
	if (r == matrix<T>::END)
		r = num_rows - 1;
	rows.push_back(r);
	view<T> *v = new view<T>(num_rows, num_cols, cols, rows, data.data());
	return *v;
}

template<class T> view<T>& matrix<T>::c_ (const vector<size_t>& cols_) {
	vector<size_t> cols, rows;
	for (size_t i=0; i < num_rows; ++i)
		rows.push_back(i);
	for (size_t j=0; j < cols_.size(); ++j) {
		if (cols_[j] < num_cols)
			cols.push_back(cols_[j]);
		else
			assert (1==0);
	}
	view<T> *v = new view<T>(num_rows, num_cols, cols, rows, data.data());
	return *v;
}
template<class T> view<T>& matrix<T>::c_ (const matrix<unsigned char>& cols_) {
	assert (cols_.rows() == 1 && cols_.cols() == num_cols);
	vector<size_t> cols, rows;
	for (size_t i=0; i < num_rows; ++i)
		rows.push_back(i);
	for (size_t j=0; j < cols_.size(); ++j)
		if (cols_(0,j) == 1)
			cols.push_back(j);
	view<T> *v = new view<T>(num_rows, num_cols, cols, rows, data.data());
	return *v;
}
template<class T> view<T>& matrix<T>::c_ (int c1, int c2) {
	assert ((c1 <= c2 && c2 < num_rows) ||
					(c1 < num_rows && c2 == matrix<T>::END) ||
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
	view<T> *v = new view<T>(num_rows, num_cols, cols, rows, data.data());
	return *v;
}
template<class T> view<T>& matrix<T>::c_ (int c) {
	assert (c < num_rows);
	vector<size_t> cols, rows;
	for (size_t i=0; i < num_rows; ++i)
		rows.push_back(i);
	if (c == matrix<T>::END)
		c = num_cols - 1;
	cols.push_back(c);
	view<T> *v = new view<T>(num_rows, num_cols, cols, rows, data.data());
	return *v;
}

template<class T> view<T>& matrix<T>::grid (const matrix<unsigned char>& rows_, const matrix<unsigned char>& cols_) {
	assert (rows_.rows() == num_rows && rows_.cols() == 1 && cols_.rows() == 1 && cols_.cols() == num_cols);
	vector<size_t> cols, rows;
	for (size_t i=0; i < rows_.size(); ++i)
		if (rows_(0,i) == 1)
			rows.push_back(i);
	for (size_t j=0; j < cols_.size(); ++j)
		if (cols_(j,0) == 1)
			cols.push_back(j);
	view<T> *v = new view<T>(num_rows, num_cols, cols, rows, data.data());
	return *v;
}

template<class T> view<T>& matrix<T>::grid (const vector<size_t>& rows_, const std::vector<size_t>& cols_) {
	assert(rows_.size() > 0 && cols_.size() > 0);
	vector<size_t> cols, rows;
	for (size_t i=0; i < rows_.size(); ++i) {
		if (rows_[i] < num_rows)
			rows.push_back(rows_[i]);
		else
			assert (1==0);
	}
	for (size_t i=0; i < cols_.size(); ++i) {
		if (cols_[i] < num_cols)
			cols.push_back(cols_[i]);
		else
			assert (1==0);
	}
	view<T> *v = new view<T>(num_rows, num_cols, cols, rows, data.data());
	return *v;
}
template<class T> view<T>& matrix<T>::area (int r1, int r2, int c1, int c2) {
	assert ((r1 <= r2 && r2 < num_rows) ||
					(r1 < num_rows && r2 == matrix<T>::END) ||
					(r1 == r2 && r2 == matrix<T>::END) ||
					(c1 <= c2 && c2 < num_cols) ||
					(c1 < num_cols && c2 == matrix<T>::END) ||
					(c1 == c2 && c2 == matrix<T>::END) );
					vector<size_t> cols, rows;
	if (r2 == matrix<T>::END)
		r2 = num_rows - 1;
	if (r1 == matrix<T>::END)
		r1 = num_rows - 1;
	for (size_t i=r1; i <= r2; ++i)
		rows.push_back(i);
	if (c2 == matrix<T>::END)
		c2 = num_cols - 1;
		if (c1 == matrix<T>::END)
			c1 = num_cols - 1;
	for (size_t j=c1; j <= c2; ++j)
		cols.push_back(j);
	view<T> *v = new view<T>(num_rows, num_cols, cols, rows, data.data());
	return *v;
}
#endif
