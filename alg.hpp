#ifndef ALGORITHMS_H_
#define ALGORITHMS_H_

#include "matrix.hpp"
#include "matrixview.hpp"

bool _sort_descending (float i, float j);
bool des_comparator (const pair<float, size_t>& l, const pair<float, size_t>& r);
bool asc_comparator (const pair<float, size_t>& l, const pair<float, size_t>& r);

matrix<float>& matrix_sort (const matrix<float>& mx, int axis=matrix<float>::SORT_FLAT, int order=matrix<float>::SORT_ASCEND);
matrix<float>& matrix_sort (const view<float>& mx, int axis=matrix<float>::SORT_FLAT, int order=matrix<float>::SORT_ASCEND);
matrix<size_t>& matrix_argsort (const view<float>& mx, int axis=matrix<float>::SORT_FLAT, int order=matrix<float>::SORT_ASCEND);
matrix<size_t>& matrix_argsort (const matrix<float>& mx, int axis=matrix<float>::SORT_FLAT, int order=matrix<float>::SORT_DESCEND);
#endif
