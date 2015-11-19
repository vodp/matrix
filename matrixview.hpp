#ifndef view_H_
#define view_H_

#include <vector>
#include <string>

using namespace std;

template<class T>
class view
{
private:
	size_t num_rows, num_cols;
	vector<size_t> strides, steps;
	T* ptr;

public:
	view (const matrix<T>& mx);
	view (size_t nrows, size_t ncols, const vector<size_t>& strides, const vector<size_t>& steps, T* ptr);
	T& operator() (size_t i, size_t j);
	T operator() (size_t i, size_t j) const;

	// dimensions
	size_t rows () const;
	size_t cols () const;
	size_t size () const;
	void size (size_t rows, size_t cols);

	// arithmetic operators
	view<T>& add (T t);
	view<T>& scale (T t);
	view<T>& power (T t);

	// reduce operators
	T max () const;
	T min () const;
	T mean () const;
	T sum () const;
	matrix<T>& max (int axis) const;
	matrix<T>& min (int axis) const;
	matrix<T>& mean (int axis) const;
	matrix<T>& sum (int axis) const;

	// fillers
	view<T>& fill (T value);
	view<T>& range (T v1, T v2);
	view<T>& randn ();

	// slicing
	view<T>& r_ (const vector<size_t>& rows);
	view<T>& r_ (const matrix<bool>& rows);
	view<T>& r_ (size_t r1, size_t r2);
	view<T>& r_ (size_t r);

	view<T>& c_ (const vector<size_t>& cols);
	view<T>& c_ (const matrix<bool>& cols);
	view<T>& c_ (size_t t1, size_t t2);
	view<T>& c_ (size_t c);

	static const size_t END = -1;
	static const size_t ALL = -2;

	matrix<T>& detach () const;
	view<T>& attach (const matrix<T>& mx);

	// assignment operator
	view<T>& operator= (const view<T>& mx);
	view<T>& operator= (const matrix<T>& mx);

	// element-wise operators
	view<T>& add (const matrix<T>& mx);
	view<T>& add (const view<T>& mx);
	view<T>& mul (const matrix<T>& mx);
	view<T>& mul (const view<T>& mx);
};

template<class T> matrix<T>& operator+ (const view<T>& mx, const view<T>& mx2);
template<class T> matrix<T>& operator- (const view<T>& mx, const view<T>& mx2);
template<class T> matrix<T>& operator* (const view<T>& mx, const view<T>& mx2);

template<class T> matrix<T>& operator+ (const view<T>& mx, const view<T>& mx2);
template<class T> matrix<T>& operator- (const view<T>& mx, const view<T>& mx2);
template<class T> matrix<T>& operator* (const view<T>& mx, const view<T>& mx2);

template<class T>
view<T>::view (const matrix<T>* mx)
	: num_rows(mx.rows()),
	  num_cols(mx.cols()),
		ptr(mx.data()),
		stridess(mx.cols(), 1),
		steps(mx.rows()m 1)
	{}

template<class T>
view (size_t nrows, size_t ncols, const vector<size_t>& strides, const vector<size_t>& steps, T* p)
	: num_rows(nrows),
	  num_cols(ncols),
		ptr(p),
		strides(strides),
		steps(steps)
	{}

template<class T>
T& view<T>::operator() (size_t i, size_t j) {
	return ptr[steps[i] * num_cols + stridess[j]];
}

template<class T>
T view<T>::operator() (size_t i, size_t j) const {
	return ptr[steps[i] * num_cols + stridess[j]];
}

template<class T> size_t view<T>::rows () { return steps.size(); }
template<class T> size_t view<T>::cols () { return strides.size(); }
template<class T> size_t view<T>::size () { return strides.size()*steps.size(); }

template<class  T> view<T>& view<T>::operator= (const view<T>& mx) {
	if (this != &mx) {
		assert(strides.size() == mx.cols() && steps.size() == mx.rows());
		for (size_t i=0; i < steps.size(); ++i)
			for (size_t j=0; j < strides.size(); ++j)
				ptr[steps[i]*num_cols + strides[j]] = mx(i,j);
	}
	return *this;
}
template<class T> view<T>& view<T>::operator= (const matrix<T>& mx) {
	assert (strides.size() == mx.cols() && steps.size() == mx.rows());
	for (size_t i=0; i < steps.size(); ++i)
		for (size_t j=0; j < strides.size(); ++j)
			ptr[steps[i]*num_cols + strides[j]] = mx(i, j);
	return *this;
}

template<class T> view<T>& view<T>::add (T t) {
	for (size_t i=0; i < steps.size(); ++i)
		for (size_t j=0; j< strides.size(); ++j)
			ptr[steps[i]*num_cols + strides[j]] += t;
	return *this;
}

template<class T> view<T>& view<T>::scale (T t) {
	for (size_t i=0; i < steps.size(); ++i)
		for (size_t j=0; j< strides.size(); ++j)
			ptr[steps[i]*num_cols + strides[j]] *= t;
	return *this;
}

template<class T> view<T>& view<T>::power (T t) {
	for (size_t i=0; i < steps.size(); ++i)
		for (size_t j=0; j< strides.size(); ++j)
			ptr[steps[i]*num_cols + strides[j]] = pow(ptr[steps[i]*num_cols + strides[j]], t);
	return *this;
}

template<class T> T view<T>::max () const {
	T max_value = numeric_limits<T>::min();
	for (size_t i=0; i < steps.size(); ++i)
		for (size_t j=0; j< strides.size(); ++j)
			if (ptr[steps[i]*num_cols + strides[j]] > max_value)
			 max_value = ptr[steps[i]*num_cols + strides[j]];
	return max_value;
}

template<class T> T view<T>::min () const {
	T min_value = numeric_limits<T>::max();
	for (size_t i=0; i < steps.size(); ++i)
		for (size_t j=0; j< strides.size(); ++j)
			if (ptr[steps[i]*num_cols + strides[j]] < min_value)
			 min_value = ptr[steps[i]*num_cols + strides[j]];
	return min_value;
}

template<class T> T view<T>::mean () const {
	T mean = 0.0;
	for (size_t i=0; i < steps.size(); ++i)
		for (size_t j=0; j< strides.size(); ++j)
			mean += ptr[steps[i]*num_cols + strides[j]];
	mean /= (strides.size()*steps.size());
	return mean;
}

template<class T> T view<T>::sum () const {
	T sum = 0.0;
	for (size_t i=0; i < steps.size(); ++i)
		for (size_t j=0; j< strides.size(); ++j)
			sum += ptr[steps[i]*num_cols + strides[j]];
	return sum;
}

template<class T> matrix<T>& view<T>::max (int axis) const {
	assert(axis == 0 || axis == 1);
	if (axis == 0) {
		matrix<T> *vector = new matrix<T>(1, strides.size());
		for (size_t j=0; j < strides.size(); ++j) {
			T max_value = numeric_limits<T>::min();
			for (size_t i=0; i < steps.size(); ++i)
				if (max_value < ptr[steps[i]*num_cols + strides[j]])
					max_value = ptr[steps[i]*num_cols + strides[j]];
			(*vector)(0, j) = max_value;
		}
		return *vector;
	} else {
		matrix<T> *vector = new matrix<T>(steps.size(), 1);
		for (size_t i=0; i < steps.size(); ++i) {
			T max_value = numeric_limits<T>::min();
			for (size_t j=0; j < strides.size(); ++j)
				if (max_value < ptr[steps[i]*num_cols + strides[j]])
					max_value = ptr[steps[i]*num_cols + strides[j]];
			(*vector)(i, 0) = max_value;
		}
		return *vector;
	}
}

template<class T> matrix<T>& view<T>::min (int axis) const {
	assert(axis == 0 || axis == 1);
	if (axis == 0) {
		matrix<T> *vector = new matrix<T>(1, strides.size());
		for (size_t j=0; j < strides.size(); ++j) {
			T min_value = numeric_limits<T>::max();
			for (size_t i=0; i < steps.size(); ++j)
				if (min_value > ptr[steps[i]*num_cols + strides[j]])
					min_value = ptr[steps[i]*num_cols + strides[j]];
			(*vector)(0, j) = min_value;
		}
		return *vector;
	} else {
		matrix<T> *vector = new matrix<T>(steps.size(), 1);
		for (size_t i=0; i < steps.size(); ++i) {
			T min_value = numeric_limits<T>::max();
			for (size_t j=0; j < strides.size(); ++j)
				if (min_value > ptr[steps[i]*num_cols + strides[j]])
					min_value = ptr[steps[i]*num_cols + strides[j]];
			(*vector)(i, 0) = min_value;
		}
	}
}

template<class T> T view<T>::mean (int axis) const {
	assert (axis == 0 || axis == 1);
	if (axis == 0) {
		matrix<T> *vector = new matrix<T>(1, strides.size());
		for (size_t j=0; j < strides.size(); ++j) {
			T mean = 0.0;
			for (size_t i=0; i < steps.size(); ++j)
				mean += ptr[steps[i]*num_cols + strides[j]];
			(*vector)(0, j) = mean / strides.size();
		}
		return *vector;
	} else {
		matrix<T> *vector = new matrix<T>(steps.size(), 1);
		for (size_t i=0; i < steps.size(); ++i) {
			T mean = 0.0;
			for (size_t j=0; j < strides.size(); ++j)
				mean += ptr[steps[i]*num_cols + strides[j]];
			(*vector)(i, 0) = mean / steps.size();
		}
		return *vector;
	}
}

template<class T> T view<T>::sum () const {
	assert (axis == 0 || axis == 1);
	if (axis == 0) {
		matrix<T> *vector = new matrix<T>(1, strides.size());
		for (size_t j=0; j < strides.size(); ++j) {
			T mean = 0.0;
			for (size_t i=0; i < steps.size(); ++j)
				mean += ptr[steps[i]*num_cols + strides[j]];
			(*vector)(0, j) = mean;
		}
		return *vector;
	} else {
		matrix<T> *vector = new matrix<T>(steps.size(), 1);
		for (size_t i=0; i < steps.size(); ++i) {
			T mean = 0.0;
			for (size_t j=0; j < strides.size(); ++j)
				mean += ptr[steps[i]*num_cols + strides[j]];
			(*vector)(i, 0) = mean;
		}
		return *vector;
	}
}

template<class T> view<T>& view<T>::fill (T value) {
	for (size_t i=0; i < steps.size(); ++i)
		for (size_t j=0; j< strides.size(); ++j)
			ptr[steps[i] * num_cols + strides[j]] = value;
	return *this;
}

template<class T> view<T>& view<T>::range(T v1, T v2) {
	T step = (v2 - v1 + 1.0)/m;
	size_t count = 0;
	for (size_t i=0; i < steps.size(); ++i)
		for (size_t j=0; j< strides.size(); ++j) {
			ptr[steps[i] * num_cols + strides[j]] = count * step + v1;
			count += 1;
		}
	return *this;
}

template<class T> view<T>& view<T>::randn () {
	srand(time(NULL));
	for (size_t i=0; i < steps.size(); ++i)
		for (size_t j=0; j< strides.size(); ++j)
			ptr[steps[i]*num_cols + strides[j]] = rand() / float(RAND_MAX);
	return *this;
}
template<class T> view<T>& view<T>::r_ (const vector<size_t>& rows) {
	assert(rows.size() == strides.size());
	vector<size_t> subrows;
	for (size_t i=0; i < rows.size(); ++i) {
		assert(rows[i] >= 0 && rows[i] < steps.size());
		subrows.push_back(steps[rows[i]]);
	}
	view<T> *subview = new view<T>(rows.size(), num_cols, strides, subrows, ptr);
	return *subview;
}
template<class T> view<T>& view<T>::r_ (const matrix<bool>& rows) {
	assert(rows.rows() == 1 && rows.cols() == strides.size());
	vector<size_t> subrows;
	for (size_t i=0; i < rows.rows(); ++i)
		if (rows(0, i) == true)
			subrows.push_back(steps[i]);
	view<T> *subview = new view<T>(rows.rows(), num_cols, strides, subrows, ptr);
	return *subview;
}
template<class T> view<T>& view<T>::r_ (size_t r1, size_t r2) {
	assert((r1 >= 0 && r1 <= r2 && r2 < steps.size()) ||
					(r1 >= 0 && r2 == view<T>::END) ||
					(r1 == view<T>::END && r2 == view<T>::END));
	vector<size_t> subrows;
	if (r1 == view<T>::END)
		r1 = r2 = steps.size() - 1;
	if (r2 == view<T>::END)
		r2 = steps.size() - 1;
	for (size_t i=r1; i <= r2; ++i)
		subrows.push_back(steps[i]);
	view<T> *subview = new view<T>(subrows.size(), num_cols, strides, subrows, ptr);
	return *subview;
}
template<class T> view<T>& view<T>::r_ (size_t r) {
	assert((r >= 0 && r < steps.size()) || r == view<T>::END);
	vector<size_t> subrow;
	subrow.push_back(steps[r]);
	if (r == view<T>::END)
		r = steps.size() - 1;
	view<T> *subview = new view<T>(1, num_cols, strides, subrow, ptr);
	return *subview;
}

template<class T> view<T>& view<T>::c_ (const vector<size_t>& cols) {
	assert(cols.size() == strides.size());
	vector<size_t> subcols;
	for (size_t i=0; i < cols.size(); ++i) {
		assert(cols[i] >= 0 && cols[i] < strides.size());
		subcols.push_back(strides[cols[i]]);
	}
	view<T> *subview = new view<T>(num_rows, cols.size(), subcols, steps, ptr);
	return *subview;
}
template<class T> view<T>& view<T>::c_ (const matrix<bool>& cols) {
	assert(cols.cols() == 1 && cols.cols() == strides.size());
	vector<size_t> subcols;
	for (size_t i=0; i < cols.cols(); ++i)
		if (cols(i, 0) == true)
			subcols.push_back(strides[i]);
	view<T> *subview = new view<T>(num_rows, cols.cols(), subcols, steps, ptr);
	return *subview;
}
template<class T> view<T>& view<T>::c_ (size_t r1, size_t r2) {
	assert((r1 >= 0 && r1 <= r2 && r2 < strides.size()) ||
					(r1 >= 0 && r2 == view<T>::END) ||
					(r1 == view<T>::END && r2 == view<T>::END));
	vector<size_t> subcols;
	if (r1 == view<T>::END)
		r1 = r2 = strides.size() - 1;
	if (r2 == view<T>::END)
		r2 = strides.size() - 1;
	for (size_t i=r1; i <= r2; ++i)
		subcols.push_back(strides[i]);
	view<T> *subview = new view<T>(num_rows, subcols.size(), subcols, steps, ptr);
	return *subview;
}
template<class T> view<T>& view<T>::c_ (size_t c) {
	assert((c >= 0 && c < strides.size()) || c == view<T>::END);
	vector<size_t> subcol;
	subcol.push_back(strides[c]);
	if (c == view<T>::END)
		c = strides.size() - 1;
	view<T> *subview = new view<T>(num_rows, 1, subcol, steps, ptr);
	return *subview;
}

template<class T> matrix<T>& view<T>::detach () const {
	assert (num_rows >=0 && num_cols >=0 && ptr != NULL &&
					strides.size() >= 0 && steps.size() >=0);
	matrix<T> *mx = new matrix<T>(steps.size(), strides.size());
	T *p = mx->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < steps.size(); ++i)
		for (size_t j=0; j < strides.size(); ++j)
			p[ix++] = ptr[steps[i]*num_cols + strides[j]];
	return *mx;
}
template<class T> view<T>& view<T>::attach (const matrix<T>& mx) {
	assert(mx.rows() == steps.size() && mx.cols() == strides.size());
	const T *p = mx.ptr();
	size_t ix = 0;
	for (size_t i=0; i < steps.size(); ++i)
		for (size_t j=0; j < strides.size(); ++j)
			ptr[steps[i] * num_cols + strides[j]] = p[ix++];
	return *this;
}

template<class T> view<T>& view<T>::add (const matrix<T>& mx) {
	assert (strides.size() == mx.cols() && steps.size() == mx.rows());
	const T *p = mx.ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j)
			ptr[steps[i] * num_cols + strides[j]] = p[ix++];
	return *this;
}
template<class T> view<T>& view<T>::add (const view<T>& mx) {
	assert (strides.size() == mx.cols() && steps.size() == mx.rows());
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j)
			ptr[steps[i] * num_cols + strides[j]] = mx(i, j);
	return *this;
}
template<class T> view<T>& view<T>::mul (const matrix<T>& mx) {
	assert (strides.size() == mx.cols() && steps.size() == mx.rows());
	const T *p = mx.ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j)
			ptr[steps[i] * num_cols + strides[j]] *= p[ix++];
	return *this;
}
template<class T> view<T>& view<T>::mul (const view<T>& mx) {
	assert (strides.size() == mx.cols() && steps.size() == mx.rows());
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j)
			ptr[steps[i] * num_cols + strides[j]] *= mx(i, j);
	return *this;
}

template<class T>
matrix<bool>&  operator== (const view<T>& mx, const T value) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) == value)
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator!= (const view<T>& mx, const T value) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) != value)
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator> (const view<T>& mx, const T value) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) > value)
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator>= (const view<T>& mx, const T value) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) >= value)
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator< (const view<T>& mx, const T value) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) < value)
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator<= (const view<T>& mx, const T value) {
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) <= value)
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}

template<class T>
matrix<bool>&  operator== (const view<T>& mx, const view<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) == mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator!= (const view<T>& mx, const view<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) != mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator> (const view<T>& mx, const view<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) > mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator>= (const view<T>& mx, const view<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) >= mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator< (const view<T>& mx, const view<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) < mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator<= (const view<T>& mx, const view<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) <= mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}

template<class T>
matrix<bool>&  operator== (const view<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) == mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator!= (const view<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) != mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator> (const view<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) > mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator>= (const view<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) >= mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator< (const view<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) < mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator<= (const view<T>& mx, const matrix<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) <= mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}

template<class T>
matrix<bool>&  operator== (const matrix<T>& mx, const view<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) == mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator!= (const matrix<T>& mx, const view<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) != mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator> (const matrix<T>& mx, const view<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) > mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator>= (const matrix<T>& mx, const view<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) >= mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator< (const matrix<T>& mx, const view<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) < mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
template<class T>
matrix<bool>&  operator<= (const matrix<T>& mx, const view<T>& mx2) {
	assert(mx.rows() == mx2.rows() && mx.cols() == mx2.cols());
	matrix<bool> *index = new matrix<bool>(mx.rows(), mx.cols());
	T *ptr = index->mutable_ptr();
	size_t ix = 0;
	for (size_t i=0; i < mx.rows(); ++i)
		for (size_t j=0; j < mx.cols(); ++j) {
			if (mx(i, j) <= mx2(i, j))
				ptr[ix++] = true;
			else
				ptr[ix++] = false;
		}
	return *index;
}
#endif
