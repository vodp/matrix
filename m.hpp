#include <vector>
#include <iostream>
using namespace std;

template<class T> class matrix {
public:
    matrix(); // default constructor
    matrix(const matrix<T>& mx); // copy constructor
    matrix(matrix<T>&& mx); // move constructor
    matrix(int rows_, int cols_);

    matrix<T>& operator= (matrix<T>&& mx); // move assignment
    matrix<T>& operator= (const matrix<T>& mx); // copy constructor

   matrix<T> mean(int axis) const;
private:
    int rows, cols;
    std::vector<T> data;
};
template<class T> matrix<T>::matrix(): rows(0), cols(0), data(0) {}
template<class T> matrix<T>::matrix (int rows_, int cols_)
    : rows(rows_), cols(cols_), data(rows * cols) {}
template<class T> matrix<T>::matrix(const matrix<T>& mx) {
    rows = mx.rows;
    cols = mx.cols;
    data = mx.data;
}
template<class T> matrix<T>::matrix(matrix<T>&& mx) {
    rows = mx.rows;
    cols = mx.cols;
    data = std::move(mx.data);
}
template<class T> matrix<T>& matrix<T>::operator= (const matrix<T>& mx) {
    if (this != &mx) {
        data.clear();
        cols = mx.cols;
        rows = mx.rows;
        data = mx.data;
    }
    return *this;
}
template<class T> matrix<T>& matrix<T>::operator= (matrix<T>&& mx) {
    if (this != &mx) {
        data.clear();
        rows = mx.rows;
        cols = mx.cols;
        data = std::move(mx.data);
    }
    return *this;
}
template<class T> matrix<T> matrix<T>::mean(int axis) const {
    matrix<T> mx(axis == 0 ? 1 : rows, axis == 1 ? cols : 1);
    // HERE compute mean vector ...
    return mx;
}
