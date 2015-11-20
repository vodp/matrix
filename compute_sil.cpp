#include "matrix.hpp"
#include "matrixview.hpp"
#include <chrono>

void load_data (const char *datafile, const char *labelfile)
{
	vector< vector<int> > clusters;
	ifstream file(labelfile);
	string line;
	size_t total_points = 0;
	while (getline (file, line))
	{
		vector<int> indices;
		stringstream ssin(line);
		while (ssin.good())
		{
			int value;
			ssin >> value;
			indices.push_back(value);
			total_points++;
		}
		clusters.push_back(indices);
	}

	// vector<int> labels(total_points);
	matrix<int> labels(total_points, 1);
	for (int i=0; i < clusters.size(); ++i)
		for (int j=0; j < clusters[i].size(); ++j)
			labels(clusters[i][j], 0) = i;

	// print out labels
	// for (int i=0; i < labels.size(); ++i)
	// 	cout << labels[i] << " ";
}

void compute_cluster_centers (const matrix<float>& X, const matrix<int>& y) {
	int num_classes = y.max() + 1;
	matrix<float>* centers = new matrix<float>(num_classes, X.cols());
	// for (int i=0; i < X.rows(); ++i) {
		// centers->r_(i) = X.r_(y == i).mean(0);
}

void compute_coefficients (const matrix<float>& X, const matrix<int>& y, const matrix<float>& centers) {
	int num_clusters = y.max() + 1;
	// group data points of the same cluster into sub-matrices
	vector<matrix<float>* > clusters;
	// for (int i=0; i < num_clusters; ++i) {
	// 	clusters.push_back(&(X.r_(y == i).detach()));
	// }
	vector<float> avg_coeffs;
	for (int i=0; i < num_clusters; ++i) {
		matrix<float> dist = cuda_pairwise_distance (*clusters[i]);
		for (int j=0; j < dist.cols(); ++j) {

		}
	}
}

void test_matrix ()
{
	// create a sample matrix
	cout << "\n";
	cout << "[Serialization test]" << endl;
	float array[] = {1.0, 2.0, 3.0, 4.0};
	matrix<float> m(array, 2, 2);
	// dump this matrix to file
	matrix<float>::dump("test.bin", m);
	// load the matrix again
	matrix<float>* m2 = matrix<float>::load("test.bin", 2, 2);
	// print it out
	cout << *m2;
	delete m2;

	cout << "\n";
	cout << "[Print test]" << endl;
	// test number 2
	int array2[] = {0, 0, 1, 1};
	matrix<int> label(array2, 4, 1);
	matrix<int>::dump("label.txt", label);
	matrix<int>* label2 = matrix<int>::load("label.txt", 4, 1);
	cout << *label2;
	delete label2;

	cout << "\n";
	cout << "[Arithmetic operators]" << endl;
	cout << "m=" << m;
	cout << "m+4 = " << m.add(4);
	cout << "m*2 = " << m.scale(2);
	cout << "m^2 = " << m.power(2.0);

	cout << "\n";
	cout << "[Assignment operators]" << endl;
	matrix<float> n;
	n = m;
	cout << "n=" << n;
	cout << "n*2 = " << n.scale(2);
	cout << "m=" << m;

	cout << "\n";
	cout << "[Copy-constructor operators]" << endl;
	matrix<float> q = m;
	cout << "q=" << q << endl;
	cout << "q*2=" << q.scale(2) << endl;
	cout << "m=" << m << endl;

	cout << endl;
	cout << "[Arithmetic operators]" << endl;
	double a[] = {1.0, 3.0, 4.0, 9.0};
	matrix<double> b(a, 2, 2);
	cout << "b = " << b << endl;
	cout << "max(b)=" << b.max() << endl;
	cout << "min(b)=" << b.min() << endl;
	cout << "sum(b)=" << b.sum() << endl;

	cout << endl;
	cout << "[Element-wise operators]" << endl;
	matrix<double> c(a, 2, 2);
	cout << "c=" << c.add(1.2) << endl;
	cout << "b + c = " << c.add(b) << endl;
	cout << "b * c = " << c.mul(b) << endl;

	// cout << endl;
	// cout << "[OpenBLAS speedup]" << endl;
	// matrix<float> x(3000, 5000), y(5000, 2000);
	// x.fill(2); y.fill(3);
	//
	// auto beg = chrono::high_resolution_clock::now();
	// fast_matrix_dot(x, y);
	// auto end = chrono::high_resolution_clock::now();
	// chrono::duration<double> dur = end - beg;
  // cout << "BLAS DOT elapsed: " << dur.count() << " seconds" << std::endl;

	// clock_t begin2 = clock();
  // fast_matrix_dot(x, y);
	// clock_t end2 = clock();
	// double elapsed_millis = double(end2 - begin2) / CLOCKS_PER_SEC;
	// cout << "BLAS DOT: " << elapsed_millis << "sec(s)" << endl;

	cout << endl;
	cout << "[pairwise_distance test]" << endl;
	matrix<float> x, y;
	x.size(3000, 256);
	y.size(4000, 256);
	auto beg = chrono::high_resolution_clock::now();
	blas_pairwise_distance(x, y);
	auto end = chrono::high_resolution_clock::now();
	auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
	// dur = end - beg;
  cout << "BLAS PDIST elapsed: " << dur.count() << " seconds" << std::endl;
}

#define TITLE(a) (cout << endl << BOLDRED << a << RESET << endl)
#define SHOW(txt,a) (cout << txt << "=" << endl << a << endl)
void test_matrix2 () {
	cout << BOLDRED << "[Arithmetic matrix]" << RESET << endl;
	matrix<float> a(3,4), b(3,4), c;
	a.randn(); b.randn();
	cout << "a=" << endl << a << endl;
	cout << "c=max(a,0)=" << endl << a.max(0) << endl;
	cout << "c=max(a,1)=" << endl << a.max(1) << endl;

	TITLE ("[Copy and References]");
	matrix<float> d = a;
	SHOW("d", d);
	a(0,0) = 0.f;
	SHOW("d", d);
	c = d;
	SHOW("c", c);
	d(0,0) = -1.f;
	SHOW("c", c);
	d = a + b;
	SHOW("d", d);
	float aa[] = {1, 2, 3, 4, 5 ,6};
	d = matrix<float>(aa, 2, 3) + matrix<float>(aa, 2, 3);
	SHOW("d", d);

	TITLE ("[Slicing and Indexing]");
	SHOW("a", a);
	view<float> va = a.r_(1,2);
	SHOW("va", va);
	view<float> va2 = a.r_(vector<size_t>({0,2}));
	SHOW("va2", va2);

	TITLE ("[Boolean operators]");
	matrix<float> e(1,4);
	e.randn();
	view<float> ve = a.c_(e == e(0,1));
	SHOW("a", a);
	SHOW("e", e);
	SHOW("ve==0", ve);
	view<float> ve2 = a.c_(0,1);
	SHOW("ve", ve2);

	matrix<float> A(5, 10);
	view<float> B = A.randn().area(1, 4, 3, 9);
	SHOW("A", A);
	SHOW("B", B);
	B.fill(0.0);
	SHOW("A", A);

	TITLE ("[View and Matrix]");
	matrix<float> C = B.detach().fill(3);
	SHOW("C", C);
	SHOW("A", A);
	B.attach(C);
	SHOW("A", A);

	A.r_(0) = matrix<float>(1, A.cols()).fill(10);
	SHOW("A", A);

	TITLE("[Detach and Attach]");
	A.size(5, 20).randn();
	SHOW("A", A);
	int ix[] = {0, 0, 1, 2, 2, 0, 3, 4, 4, 4, 3, 4, 2, 2, 1, 0, 0, 3, 3, 3};
	matrix<int> y(ix, 1, 20);
	vector<matrix<float>* > subs;
	for (int i=0; i < 5; ++i)
		subs.push_back(&(A.c_(y == i).detach()));
	// print them out
	for (int i=0; i < 5; ++i)
		SHOW("-", *subs[i]);
}

void test_cuda() {
	cout << endl;
	cout << "[CuBLAS test]" << endl;
	float yy[] = {1, 3, 5, 2, 4, 6};
	matrix<float> x(4, 2), y(yy, 2, 3);
	x.range(1, 8);
	cout << "x=" << endl << x << endl;
	// y.range(3, 6);
	cout << "y=" << endl << y << endl;

	int dev_id = 0;
	cuda_init(dev_id);
	auto beg = chrono::high_resolution_clock::now();
	cout << "z=" << endl << cuda_matrix_dot(x, y) << endl;
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double> dur = end - beg;
  cout << "CuBLAS DOT elapsed: " << dur.count() << " seconds" << std::endl;
}

void test_cuda_pdist() {
	cout << endl;
	cout << "[CuBLAS pairwise_distance]" << endl;
	matrix<float> x(4000, 2000), y(3000, 2000);
	// x.range(1, 8);
	// cout << "x=" << endl << x << endl;
	// y.range(1, 6);
	// cout << "y=" << endl << y << endl;
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

void test_cluster() {
	load_data("test.bin", "test.txt");
}

int main()
{
	// test_cuda();
	// test_cuda_pdist();
	test_matrix2();
}
