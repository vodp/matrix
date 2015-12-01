#include "matrix.hpp"
#include "matrixview.hpp"

bool _sort_descending (float i, float j) {
	return i > j;
}
bool asc_comparator (const pair<float, size_t>& l, const pair<float, size_t>& r) {
	return l.first < r.first;
}
bool des_comparator (const pair<float, size_t>& l, const pair<float, size_t>& r) {
	return l.first > r.first;
}
matrix<float>& matrix_sort (const matrix<float>& mx, int axis, int order) {
	assert (axis == matrix<float>::SORT_FLAT ||
		 			axis == matrix<float>::SORT_COLS ||
					axis == matrix<float>::SORT_ROWS);
	matrix<float>* sorted = new matrix<float>(mx.rows(), mx.cols());
	*sorted = mx;
	if (axis == matrix<float>::SORT_FLAT) {
		// sort as if the matrix is flattened
		float *ptr = sorted->mutable_ptr();
		if (order == matrix<float>::SORT_ASCEND)
			sort (ptr, ptr + sorted->size());
		else
			sort (ptr, ptr + sorted->size(), _sort_descending);
	} else if (axis == matrix<float>::SORT_ROWS) {
		for (size_t i=0; i < sorted->rows(); ++i) {
			float *ptr = sorted->mutable_ptr();
			if (order == matrix<float>::SORT_ASCEND)
				sort (ptr + i*sorted->cols(), ptr + (i+1)*sorted->cols());
			else
				sort (ptr + i*sorted->cols(), ptr + (i+1)*sorted->cols(), _sort_descending);
		}
	} else {
		sorted->t();
		for (size_t i=0; i < sorted->rows(); ++i) {
			float *ptr = sorted->mutable_ptr();
			if (order == matrix<float>::SORT_ASCEND)
				sort (ptr + i*sorted->cols(), ptr + (i+1)*sorted->cols());
			else
				sort (ptr + i*sorted->cols(), ptr + (i+1)*sorted->cols(), _sort_descending);
		}
		sorted->t();
	}
	return *sorted;
}
matrix<size_t>& matrix_argsort (const matrix<float>& mx, int axis, int order) {
	assert (axis == matrix<float>::SORT_FLAT ||
					axis == matrix<float>::SORT_COLS ||
					axis == matrix<float>::SORT_ROWS);
	vector<std::pair<float, size_t> > elems;
	if (axis == matrix<float>::SORT_FLAT) {
		const float* ptr = mx.ptr();
		size_t m = mx.size();
		for (size_t i=0; i < m; ++i)
			elems.push_back(std::make_pair(ptr[i], i));
	} else if (axis == matrix<float>::SORT_COLS) {
		matrix<float> sorted = mx;
		sorted.t();
		const float* ptr = sorted.ptr();
		size_t count = 0;
		for (size_t j=0; j < sorted.cols(); ++j)
			for (size_t i=0; i < sorted.rows(); ++i)
				elems.push_back(std::make_pair(ptr[count++], i));
	} else {
		const float* ptr = mx.ptr();
		size_t count = 0;
		for (size_t i=0; i < mx.rows(); ++i)
			for (size_t j=0; j < mx.cols(); ++j)
				elems.push_back(std::make_pair(ptr[count++], j));
	}
	// sorting
	if (axis == matrix<float>::SORT_FLAT) {
		if (order  == matrix<float>::SORT_ASCEND)
			std::sort (elems.begin(), elems.end(), asc_comparator);
		else
			std::sort (elems.begin(), elems.end(), des_comparator);
	} else if (axis == matrix<float>::SORT_ROWS){
		for (size_t i=0; i < mx.rows() - 1; ++i) {
			if (order == matrix<float>::SORT_ASCEND)
				std::sort (elems.begin() + i*mx.cols(), elems.begin() + (i+1)*mx.cols(), asc_comparator);
			else
				std::sort (elems.begin() + i*mx.cols(), elems.begin() + (i+1)*mx.cols(), des_comparator);
		}
	} else {
		for (size_t i=0; i < mx.cols() - 1; ++i) {
			if (order == matrix<float>::SORT_ASCEND)
				std::sort (elems.begin() + i*mx.rows(), elems.begin() + (i+1)*mx.rows(), asc_comparator);
			else
				std::sort (elems.begin() + i*mx.rows(), elems.begin() + (i+1)*mx.rows(), des_comparator);
		}
	}
	matrix<size_t>* index = new matrix<size_t>(mx.rows(), mx.cols());
	size_t *ptri = index->mutable_ptr();
	for (size_t i=0; i < elems.size(); ++i)
		ptri[i] = elems[i].second;
	if (axis == matrix<float>::SORT_COLS)
		index->t();
	return *index;
}
matrix<size_t>& matrix_argsort (const view<float>& vx, int axis, int order) {
	assert (axis == matrix<float>::SORT_FLAT ||
					axis == matrix<float>::SORT_COLS ||
					axis == matrix<float>::SORT_ROWS);
	vector<std::pair<float, size_t> > elems;
	matrix<float> mx;
	mx = vx;
	if (axis == matrix<float>::SORT_FLAT) {
		const float* ptr = mx.ptr();
		size_t m = mx.size();
		for (size_t i=0; i < m; ++i)
			elems.push_back(std::make_pair(ptr[i], i));
	} else if (axis == matrix<float>::SORT_COLS) {
		mx.t();
		const float* ptr = mx.ptr();
		size_t count = 0;
		for (size_t j=0; j < mx.cols(); ++j)
			for (size_t i=0; i < mx.rows(); ++i)
				elems.push_back(std::make_pair(ptr[count++], i));
	} else {
		const float* ptr = mx.ptr();
		size_t count = 0;
		for (size_t i=0; i < mx.rows(); ++i)
			for (size_t j=0; j < mx.cols(); ++j)
				elems.push_back(std::make_pair(ptr[count++], j));
	}
	// sorting
	if (axis == matrix<float>::SORT_FLAT) {
		if (order  == matrix<float>::SORT_ASCEND)
			std::sort (elems.begin(), elems.end(), asc_comparator);
		else
			std::sort (elems.begin(), elems.end(), des_comparator);
	} else {
		for (size_t i=0; i < mx.rows() - 1; ++i) {
			if (order == matrix<float>::SORT_ASCEND)
				std::sort (elems.begin() + i*mx.cols(), elems.begin() + (i+1)*mx.cols(), asc_comparator);
			else
				std::sort (elems.begin() + i*mx.cols(), elems.begin() + (i+1)*mx.cols(), des_comparator);
		}
	}
	matrix<size_t>* index = new matrix<size_t>(mx.rows(), mx.cols());
	size_t *ptri = index->mutable_ptr();
	for (size_t i=0; i < elems.size(); ++i)
		ptri[i] = elems[i].second;
	if (axis == matrix<float>::SORT_COLS)
		index->t();
	return *index;
}
matrix<float>& matrix_sort (const view<float>& mx, int axis, int order) {
	assert (axis >= -1 && axis <= 1);
	matrix<float>* sorted = new matrix<float>(mx.rows(), mx.cols());
	*sorted = mx;
	if (axis == matrix<float>::SORT_FLAT) {
		// sort as if the matrix is flattened
		float *ptr = sorted->mutable_ptr();
		if (order == matrix<float>::SORT_ASCEND)
			sort (ptr, ptr + sorted->size());
		else
			sort (ptr, ptr + sorted->size(), _sort_descending);
	} else if (axis == matrix<float>::SORT_ROWS) {
		for (size_t i=0; i < sorted->rows(); ++i) {
			float *ptr = sorted->mutable_ptr();
			if (order == matrix<float>::SORT_ASCEND)
				sort (ptr + i*sorted->cols(), ptr + (i+1)*sorted->cols());
			else
				sort (ptr + i*sorted->cols(), ptr + (i+1)*sorted->cols(), _sort_descending);
		}
	} else {
		sorted->t();
		for (size_t i=0; i < sorted->rows(); ++i) {
			float *ptr = sorted->mutable_ptr();
			if (order == matrix<float>::SORT_ASCEND)
				sort (ptr + i*sorted->cols(), ptr + (i+1)*sorted->cols());
			else
				sort (ptr + i*sorted->cols(), ptr + (i+1)*sorted->cols(), _sort_descending);
		}
		sorted->t();
	}
	return *sorted;
}
