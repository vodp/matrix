#include "matrix.hpp"
#include "matrixview.hpp"
#include "alg.hpp"
#include <functional>
#include <cctype>
#include <locale>
#include <chrono>

void test_matrix () {
	TITLE("[Serialization test]");
	float array[] = {1.0, 2.0, 3.0, 4.0};
	matrix<float> m(array, 2, 2);
	matrix<float>::dump("test.bin", m);
	matrix<float> m2 = matrix<float>::load("test.bin", 2, 2);

	TITLE("[Arithmetic operators]");
	SHOW("m", m);
	SHOW("m+4", m.add(4.0));
	SHOW("m*2", m.mul(2.0));
	SHOW("m^2", m.power(2.0));

	matrix<float> a(3,4);
	a.randn();
	SHOW("a", a);
	SHOW("max(a,0)", a.max(0));
	SHOW("max(a,1)", a.max(1));

	TITLE("[Assignment operators]");
	matrix<float> n;
	n = m;
	SHOW("n", n);
	SHOW("n*2", n.mul(2.0));
	SHOW("m", m);

	TITLE("[Copy-constructor operators]");
	matrix<float> q = m;
	SHOW("q", q);
	SHOW("q*2", q);
	SHOW("m", m);

	TITLE("[Arithmetic operators]");
	double bb[] = {1.0, 3.0, 4.0, 9.0};
	matrix<double> b(bb, 2, 2);
	SHOW("b", b);
	SHOW("max(b)", b.max());
	SHOW("min(b)", b.min());
	SHOW("sum(b)", b.sum());

	TITLE("[Element-wise operators]");
	matrix<double> c(bb, 2, 2);
	SHOW("c", c.add(1.2));
	SHOW("b + c", c.add(b));
	SHOW("b * c", c.mul(b));

	TITLE ("[Slicing and Indexing]");
	SHOW("a", a);
	view<float> va = a.r_(1,2);
	SHOW("va", va);
	view<float> va2 = a.r_(vector<size_t>({0,2}));
	SHOW("va2", va2);
	matrix<float> A(5, 10);
	view<float> B = A.randn().rc_(1, 4, 3, 9);
	SHOW("A", A);
	SHOW("B", B);
	B.fill(0.0);
	SHOW("A", A);
	A.r_(0) = matrix<float>(1, A.cols()).fill(10);
	SHOW("A", A);

	TITLE("[Fillers]");
	matrix<float> e(1,4);
	e.randn();
	SHOW("e", e);
	e.fill(0.0);
	SHOW("e", e);

	TITLE ("[View and Matrix]");
	view<float> ve = a.c_(e == e(0,1));
	SHOW("a", a);
	SHOW("e", e);
	SHOW("ve==0", ve);
	view<float> ve2 = a.c_(0,1);
	SHOW("ve", ve2);

	TITLE ("[Detach and Attach]");
	A.size(5,7);
	A.randn();
	SHOW("A", A);
	int ix[] = {0, 0, 1, 1, 2};
	matrix<int> y(ix, 1,5);
	SHOW("y", y);
	view<float> aa = A.r_(y == 0);
	SHOW("a", aa);
	matrix<unsigned char> duma = y == 0;
	SHOW("y == 0", duma);
	matrix<float> C = aa.detach();
	SHOW("B", C);
}
void test_openblas() {
	TITLE("[OpenBLAS speedup]");
	matrix<float> x(3000, 5000), y(5000, 2000);
	x.fill(2); y.fill(3);

	auto beg = chrono::high_resolution_clock::now();
	blas_matrix_dot(x, y);
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double> dur = end - beg;
  cout << "BLAS DOT elapsed: " << dur.count() << " seconds" << std::endl;
}
void test_openblash_pdist() {
	TITLE("[Pairwise distance Test]");
	matrix<float> x, y;
	x.size(3000, 256);
	y.size(4000, 256);
	auto beg = chrono::high_resolution_clock::now();
	blas_pairwise_distance(x, y);
	auto end = chrono::high_resolution_clock::now();
	auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
	cout << "BLAS PDIST elapsed: " << dur.count() << " seconds" << std::endl;
}
void test_sorting () {
	matrix<float> A(4, 4);
	A.randn();
	TITLE("[Sorting test]");
	SHOW("A", A);
	matrix<float> B = matrix_sort (A, matrix<float>::SORT_ROWS, matrix<float>::SORT_ASCEND);
	SHOW("sort(A, row)", B);
	matrix<size_t> C = matrix_argsort (A, matrix<float>::SORT_ROWS, matrix<float>::SORT_ASCEND);
	SHOW("arg(A, row)", C);

	B = matrix_sort (A, matrix<float>::SORT_COLS, matrix<float>::SORT_ASCEND);
	SHOW("sort(A, col)", B);
	C = matrix_argsort (A, matrix<float>::SORT_COLS, matrix<float>::SORT_ASCEND);
	SHOW("arg(A, col)", C);

	B = matrix_sort (A, matrix<float>::SORT_ROWS, matrix<float>::SORT_DESCEND);
	SHOW("sort(B, row, desc)", B);
	C = matrix_argsort (A, matrix<float>::SORT_ROWS, matrix<float>::SORT_DESCEND);
	SHOW("arg(B, row, desc)", C);

	B = matrix_sort (A, matrix<float>::SORT_COLS, matrix<float>::SORT_DESCEND);
	SHOW("sort(B, col, desc)", B);
	C = matrix_argsort (A, matrix<float>::SORT_COLS, matrix<float>::SORT_DESCEND);
	SHOW("arg(B, col, desc)", C);

	view<float> a = A.rc_(0, 2, 0, 2);
	SHOW("a", a);
	matrix<float> b = matrix_sort (a, matrix<float>::SORT_ROWS, matrix<float>::SORT_ASCEND);
	SHOW("sort(a, row, asc)", b);
	matrix<size_t> c = matrix_argsort (a, matrix<float>::SORT_ROWS, matrix<float>::SORT_ASCEND);
	SHOW("arg(a, row, asc)", c);

	b = matrix_sort (a, matrix<float>::SORT_COLS, matrix<float>::SORT_ASCEND);
	SHOW("sort(a, row, asc)", b);
	c = matrix_argsort (a, matrix<float>::SORT_COLS, matrix<float>::SORT_ASCEND);
	SHOW("arg(a, row, asc)", c);

	b = matrix_sort (a, matrix<float>::SORT_ROWS, matrix<float>::SORT_DESCEND);
	SHOW("sort(a, row, des)", b);
	c = matrix_argsort (a, matrix<float>::SORT_ROWS, matrix<float>::SORT_DESCEND);
	SHOW("arg(a, row, des)", c);
}
void test_cuda() {
	TITLE("[CuBLAS test]");
	float yy[] = {1, 3, 5, 2, 4, 6};
	matrix<float> x(4, 2), y(yy, 2, 3);
	x.range(1, 8);
	SHOW("x", x);
	SHOW("y", y);

	int dev_id = 0;
	cuda_init(dev_id);
	SHOW("z", cuda_matrix_dot(x, y));
}
void test_cuda_pdist() {
	TITLE("[CuBLAS pairwise_distance]");
	matrix<float> x(4000, 2000), y(3000, 2000);
	int dev_id = 0;
	cuda_init(dev_id);
	auto beg = chrono::high_resolution_clock::now();
	cuda_pairwise_distance(x, y);
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double> dur = end - beg;
  cout << "CuBLAS PDIST elapsed: " << dur.count() << " seconds" << std::endl;

	beg = chrono::high_resolution_clock::now();
	cuda_pairwise_distance(x);
	end = chrono::high_resolution_clock::now();
	dur = end - beg;
  cout << "CuBLAS PDIST elapsed: " << dur.count() << " seconds" << std::endl;
}
int main() {
	test_matrix();
	test_sorting();
	test_openblas();
	test_openblash_pdist();
	test_cuda();
	test_cuda_pdist();
}
