#include "matrix.hpp"
#include "matrixview.hpp"
#include "alg.hpp"
#include <functional>
#include <cctype>
#include <locale>
#include <chrono>

#define TITLE(a) (cout << endl << BOLDRED << a << RESET << endl)
#define SHOW(txt,a) (cout << txt << "=" << endl << a << endl)

// trim from start
static inline std::string &ltrim(std::string &s) {
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
	return s;
}
// trim from end
static inline std::string &rtrim(std::string &s) {
	s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
  return s;
}
// trim from both ends
static inline std::string &trim(std::string &s) {
	return ltrim(rtrim(s));
}
static inline bool do_file_exist (const std::string& name) {
	if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else
        return false;
}

matrix<int>& load_labels (const char *labelfile) {
	vector< vector<int> > clusters;
	ifstream file(labelfile);
	string line;
	size_t total_points = 0;
	while (getline (file, line)) {
		vector<int> indices;
		line = trim(line);
		stringstream ssin(line);
		while (ssin.good()) {
			int value;
			ssin >> value;
			indices.push_back(value);
			total_points++;
		}
		clusters.push_back(indices);
	}
	matrix<int>* labels = new matrix<int>(total_points, 1);
	for (int i=0; i < clusters.size(); ++i)
		for (int j=0; j < clusters[i].size(); ++j)
			(*labels)(clusters[i][j], 0) = i;
	return *labels;
}

matrix<float>& compute_cluster_centers (matrix<float>& X, matrix<int>& y) {
	assert(X.rows() == y.size());
	int num_classes = y.max() + 1;
	matrix<float>* centers = new matrix<float>(num_classes, X.cols());
	size_t m = X.rows();
	for (int i=0; i < num_classes; ++i) {
		// cout << "computing cluster center #" << i << endl;
		centers->r_(i) = X.r_(y == i).mean(0);
	}
	return *centers;
}
matrix<float>& compute_coefficients (matrix<float>& X, matrix<int>& y, matrix<float>& centers) {
	int num_clusters = y.max() + 1;
	// copy original data into submatrices to avoid further computational cost
	// vector<matrix<float>* > clusters;
	// cout << "Grouping cluster data..." << endl;
	// for (int i=0; i < num_clusters; ++i) {
	// 	clusters.push_back(&(X.r_(y == i).detach()));
	// }
	// delete &X;
	matrix<float> *avg_coeffs = new matrix<float>(1, num_clusters);
	for (int i=0; i < num_clusters; ++i) {
		cout << "Computing cluster #" << i << "..." << endl;
		matrix<unsigned char>& ixx = y == i;
		view<float>& cluster_view = X.r_(y == i);
		matrix<float>& thiscluster = cluster_view.detach();
		matrix<float>& inter_dist = cuda_pairwise_distance (thiscluster, centers);
		matrix<size_t>& sorted_ix = matrix_argsort(inter_dist, matrix<size_t>::SORT_ROWS);
		matrix<size_t> nearest_clusters(1, thiscluster.rows());
		for (int j=0; j < sorted_ix.rows(); ++j) {
			if (sorted_ix(j,0) != i)
				nearest_clusters(0,j) = sorted_ix(j, 0);
			else
				nearest_clusters(0,j) = sorted_ix(j, 1);
		}
		// group points having common nearest clusters and compute silhouette coefficients
		vector<size_t> unique_nearest_clusters(nearest_clusters.ptr(), nearest_clusters.ptr() + nearest_clusters.size());
		std::sort (unique_nearest_clusters.begin(), unique_nearest_clusters.end());
		vector<size_t>::iterator it = std::unique (unique_nearest_clusters.begin(), unique_nearest_clusters.end());
		unique_nearest_clusters.resize(std::distance(unique_nearest_clusters.begin(), it) );

		int count = 0;
		matrix<float>& intra_dist = cuda_pairwise_distance (thiscluster);
		matrix<float> coeffs(1, nearest_clusters.size());

		for (size_t k=0; k < unique_nearest_clusters.size(); ++k) {
			size_t next_cluster = unique_nearest_clusters[k];
			matrix<unsigned char>& ix = y == (int)next_cluster;
			view<float>& subview = X.r_(ix);
			matrix<float>& subcluster = subview.detach();

			matrix<unsigned char>& bool_ix = nearest_clusters == next_cluster;
			matrix<float>& vecs = thiscluster.r_(bool_ix).detach();
			matrix<float>& subinter = cuda_pairwise_distance(vecs, subcluster);
			matrix<float>& mean_b = subinter.mean(1);
			view<float>& subintra = intra_dist.r_(bool_ix);
			matrix<float>& mean_a = subintra.mean(1);
			assert(mean_a.size() == mean_b.size() && vecs.rows() == mean_a.size());
			for (int j=0; j < mean_a.rows(); ++j) {
				float coeff = (mean_b(j,0) - mean_a(j,0))/std::max(mean_b(j,0), mean_a(j,0));
				coeffs(0, count++) = coeff;
			}
			delete &subcluster;
			delete &bool_ix;
			delete &vecs;
			delete &mean_b;
			delete &mean_a;
			delete &ix;
			delete &subview;
			delete &subinter;
			delete &subintra;
		}
		assert(count == coeffs.size());
		(*avg_coeffs)(0, i) = coeffs.mean();
		cout << "coefficient #" << i << ": " << (*avg_coeffs)(0,i) << endl;
		delete &thiscluster;
		delete &intra_dist;
		delete &inter_dist;
		delete &sorted_ix;
		delete &cluster_view;
		delete &ixx;
	}
	return (*avg_coeffs);
}
void csil () {
	cout << "Loading features..." << endl;
	matrix<float>& X = matrix<float>::load("../dat/20m_signatures_random.caffe.256", 18389592, 256);
	X = X.c_(0, 16).detach();
	cout << "...has dimension (" << X.rows() << ", " << X.cols() << ")" << endl;
	cout << "Loading labels..." << endl;
	matrix<int>& y = load_labels("../dat/cluster_20msig_5kcenter_random.lst");
	cout << "...has dimension (" << y.rows() << ", " << y.cols() << ")" << endl;
	view<int> yy = y.r_(std::vector<size_t>({5176, 10577, 10940, 11167, 15510, 15914}));
	SHOW("y[:6]", yy);
	matrix<float> centers;
	if (do_file_exist(string("../dat/centers.16"))) {
		cout << "Loading centers..." << endl;
		centers = matrix<float>::load("../dat/centers.16", 5000, 16);
	} else {
		cout << "Computing cluster centers..." << endl;
		centers = compute_cluster_centers (X, y);
		cout << "Saving cluster centers..." << endl;
		matrix<float>::dump("../dat/centers.16", centers);
	}
	cout << "Computing Silhouette coefficients..." << endl;
	matrix<float>& coeffs = compute_coefficients (X, y, centers);
	matrix<float>::dump("../dat/coeffs.bin", coeffs);
	cout << "Dumped results to ../dat/coeffs.bin. DONE.";
	delete &X;
	delete &y;
	delete &coeffs;
}
// void select_training_data () {
// 	matrix<float>& coeffs = matrix<float>::load("../dat/coeffs.bin", 5000, 1);
// 	matrix<size_t>& top_clusters = matrix_argsort(coeffs);
// 	view<size_t>& top1k = top_clusters.c_(0,1000);
// 	cout << "The largest Silhouette coefficient = " << coeffs(top1k(0,0), 0) << endl;
// 	cout << "The smallest Silhouette coefficient = " << coeffs(top1k(matrix<size_t>::END,0), 0) << endl;
// 	cout << "Loading labels..." << endl;
// 	matrix<int>&  y = load_labels("../dat/cluster_20msig_5kcenter_random.lst");
// 	ofstream out_file("../dat/train.txt", ios::out);
// 	for (size_t i=0; i < 1000; i++) {
// 		matrix<unsigned char>& ix = y == top1k(i,0);
// 		view<int>& suby = y.r_(ix);
// 		for (size_t j=0; j < suby.size(); ++j)
// 			out_file << suby(j,0) << " " << top1k(i,0) << endl;
// 		delete &suby;
// 		delete &ix;
// 		delete &suby;
// 	}
// 	out_file.close();
// 	delete &coeffs;
// 	delete &top_clusters;
// 	delete &top1k;
// }

void csil_sanitycheck () {
	cout << "Loading features..." << endl;
	matrix<float> X = matrix<float>::load("../dat/MNIST.dat", 10000, 5);
	cout << "...has dimension (" << X.rows() << ", " << X.cols() << ")" << endl;
	cout << "Loading labels..." << endl;
	matrix<int> y = matrix<int>::load("../dat/MNIST.label", 10000, 1);
	cout << "...has dimension (" << y.rows() << ", " << y.cols() << ")" << endl;
	matrix<float> centers;
	if (do_file_exist(string("../dat/centers.5"))) {
		cout << "Loading centers..." << endl;
		centers = matrix<float>::load("../dat/centers.5", 10, 5);
	} else {
		cout << "Computing cluster centers..." << endl;
		centers = compute_cluster_centers (X, y);
		cout << "Saving cluster centers..." << endl;
		matrix<float>::dump("../dat/centers.16", centers);
	}
	cout << "Computing Silhouette coefficients..." << endl;
	matrix<float> coeffs = compute_coefficients (X, y, centers);
	matrix<float>::dump("../dat/coeffs.bin", coeffs);
	cout << "Dumped results to ../dat/coeffs.bin. DONE.";
}

void test_matrix () {
	// create a sample matrix
	cout << "\n";
	cout << "[Serialization test]" << endl;
	float array[] = {1.0, 2.0, 3.0, 4.0};
	matrix<float> m(array, 2, 2);
	// dump this matrix to file
	matrix<float>::dump("test.bin", m);
	// load the matrix again
	matrix<float> m2 = matrix<float>::load("test.bin", 2, 2);
	// print it out
	cout << m2;

	cout << "\n";
	cout << "[Print test]" << endl;
	// test number 2
	int array2[] = {0, 0, 1, 1};
	matrix<int> label(array2, 4, 1);
	matrix<int>::dump("label.txt", label);
	matrix<int> label2 = matrix<int>::load("label.txt", 4, 1);
	cout << label2;

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

	view<float> a = A.area(0, 2, 0, 2);
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

void test_detach() {
	matrix<float> A(5,7);
	A.randn();
	SHOW("A", A);
	int ix[] = {0, 0, 1, 1, 2};
	matrix<int> y(ix, 1,5);
	SHOW("y", y);
	view<float> a = A.r_(y == 0);
	SHOW("a", a);
	matrix<unsigned char> duma = y == 0;
	SHOW("y == 0", duma);
	matrix<float> B = a.detach();
	SHOW("B", B);
	// matrix<float>C = B.r_(2,3).detach();
	// SHOW("C", C);
}

void test_c11() {
	matrix<float> a(4, 5);
	a.fill(1.f);
	matrix<unsigned char> ix = a == 0.f;
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
	test_c11();
}
