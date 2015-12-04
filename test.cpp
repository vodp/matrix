#include "matrix.hpp"
#include "matrixview.hpp"
#include "alg.hpp"
#include <functional>
#include <cctype>
#include <locale>
#include <chrono>

#define TITLE(a) (cout << endl << BOLDRED << a << RESET << endl)
#define SHOW(txt,a) (cout << txt << "=" << endl << a << endl)

// void test_c11() {
// 	matrix<float> a(4, 5), b(4, 5);
// 	a.randn();
// 	b.fill(5.0f);
// 	matrix<float> c = std::move((a * 3.f + b).scale(2.0f));
// 	SHOW("a", a);
// 	SHOW("b", b);
// 	SHOW("c", c);
// 	matrix<float> e(5,2);
// 	e.fill(2.0f);
// 	matrix<float> d = c.dot(e);
// 	SHOW("e", e);
// 	SHOW("d", d);
// 	d(0,0) = 0.0f;
// 	SHOW("c", c);
// 	matrix<float> f(c);
// 	f.randn();
// 	SHOW("f", f);
// 	SHOW("c", c);
// }
matrix<float> f() {
	matrix<float> a(1,2);
	return a;
}
matrix<float> g() {
	matrix<float> *b = new matrix<float>(1,2);
	return *b;
}
matrix<float>& k() {
	matrix<float> *b = new matrix<float>(1,2);
	return *b;
}

void test_3() {
	matrix<float> a;
	a = f();
	cout << "--" << endl;
	a = g();
	cout << "--" << endl;
	a = a.mean(1);
}
void test_2() {
	matrix<float> x(2000, 3000), y(2000, 3000);
	x.fill(2.0f); y.fill(3.0f);
	view<float> zz = x.r_(y.c_(0,1) == 3.0f);
	cout << "--" << endl;
	matrix<float> z(1,2);
	cout << "##" << endl;
	z = x.mean(1);
	// matrix<float> z = x.r_(y.c_(0,1) == 3.0f).mean(1);
	// z = z.scale(2.0f);
}
int main()
{
	// test_cuda();
	// test_cuda_pdist();
	// test_matrix2();
	// test_sorting();
	// csil();
	// csil_sanitycheck();
	// test_detach();
	test_3();
}
