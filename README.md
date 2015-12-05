# matrix
A minimal matrix utility written in C++11 template. I use this `matrix` to speed up some heavy calculation such as matrix dot-product and Euclidean pairwise distances by using OpenMP, OpenBLAS and CUDA. To reduce memory copy overhead and obscure pointer operations which is conventionally encountered in C and C++98 or C++03, I adopt the new standard C++11. This standard comes up with efficient compiler-level optimization such as copy elision, move constructor, and move assignment. You mostly do not have to use dynamic memory with this library so that memory release is automatically done on the behalf of the compiler.

Since `matrix` aim to work with very large matrices, its operations should be memory-saving and restrain from data-copying around. There are two main objects which are `matrix<T>` and `view<T>`. Matrix is the only object the stores raw data, view is a reference to a part of matrix's data (based on various slicing methods) therefore it does not contain independent data. A view that is attached to a particular matrix, then all operation with that view will affect to the data in matrix. View can be detached from its owner matrix by method `detach()` in order to create another sub-matrix. This sub-matrix can be attached again afterward. 

In other word, if you have a very large matrix but you just work with part of it at a particular time, the you can create a "cheap" view to work with. If operations must not affect the original data, then `detach()` the view from the matrix. Since view does not bring raw data, memory is not wasted.

However notice that to get very fast matrix multiplication based on BLAS or CUDA, you must not must view but matrix. The reason is simple, continous memory storage is the key to those operations.

### Prerequisites
0. Linux (I have not built on Windows or OS)
1. `gcc` and `g++` version >= 4.8.*
2. OpenMP is supported by the OS
3. *(Optional)* Install OpenBLAS
4. *(Optional)* Install CUDA Toolkit version >= 7.0

### Compatibility
In order to get most benefit from `matrix`, you have to compile it with g++ 4.8.* which supports C++11. CUDA Toolkit from 7.0 is known to officially support C++11 standard.

### Compile
This library is minimal and being a template like Boost, so you do not need to compile it. Just put its source files into your projects compile together. See the `Makefile` for more details.

### Howto use
Below is un incomplete list of operations with `matrix`. More examples could be found in file `test.cpp`.
To declare a matrix of type `float` of dimension 2x2:

    matrix<float> m(2, 2);
    // or 
    float array[] = {1.f, 2.f, 3.f, 4.f};
    matrix<float> m(array, 2, 2);
    // or 
    std::vector data({1.f, 2.f, 3.f, 4.f});
    matrix<float> m(data, 2, 2);

#### Get capacity

    // number of rows
    size_t r = m.rows();
    // number of cols
    size_t c = m.cols(); 
    // matrix size
    matrix<size_t> shape = m.shape();
    // size of particular dimension
    size_t d0 = m.shape(0); // rows
    size_t d1 = m.shape(1); // cols

#### Set capacity

    m.size(10, 20); // change size of matrix from 2x2 to 10x20

#### Fillers

    m.fill(-1.f); // fill the whole matrix with 1.f
    m.randn(); // put random values in [0, 1)

#### Deduction

    // Get maximum value
    float v = m.max();
    // Get a vector of maximum value at each row
    matrix<float> v = m.max(1);

#### Slicing

    // get element (i,j)
    float v = m(i,j);
    // set element at (i,j)
    m(i,j) = 1000.f;
    // get row i-th
    view<float> vx = m.r_(i);
