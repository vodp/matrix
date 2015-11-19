#include "matrix.hpp"
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

	vector<int> labels(total_points);
	for (int i=0; i < clusters.size(); ++i)
		for (int j=0; j < clusters[i].size(); ++j)
		{
			labels[clusters[i][j]] = i;
		}

	// print out labels
	for (int i=0; i < labels.size(); ++i)
		cout << labels[i] << " ";
}

void test_matrix ()
{
	// create a sample matrix
	cout << "\n";
	cout << "[Serialization test]" << endl;
	float array[] = {1.0, 2.0, 3.0, 4.0};
	matrix<float> m(array, 2, 2);
	// dump this matrix to file
	matrix<float>::dump_matrix("test.bin", m);
	// load the matrix again
	matrix<float>* m2 = matrix<float>::load_matrix("test.bin", 2, 2);
	// print it out
	cout << *m2;
	delete m2;

	cout << "\n";
	cout << "[Print test]" << endl;
	// test number 2
	int array2[] = {0, 0, 1, 1};
	matrix<int> label(array2, 4, 1);
	matrix<int>::dump_matrix("label.txt", label);
	matrix<int>* label2 = matrix<int>::load_matrix("label.txt", 4, 1);
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
	cout << "mean(b)=" << b.mean() << endl;

	cout << endl;
	cout << "[Element-wise operators]" << endl;
	matrix<double> c(a, 2, 2);
	cout << "c=" << c.add(1.2) << endl;
	cout << "b + c = " << c.add(b) << endl;
	cout << "b * c = " << c.mul(b) << endl;

	cout << endl;
	cout << "[OpenBLAS speedup]" << endl;
	matrix<float> x(3000, 5000), y(5000, 2000);
	x.fill(2); y.fill(3);

	auto beg = chrono::high_resolution_clock::now();
	fast_matrix_dot(x, y);
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double> dur = end - beg;
  cout << "BLAS DOT elapsed: " << dur.count() << " seconds" << std::endl;

	// clock_t begin2 = clock();
  // fast_matrix_dot(x, y);
	// clock_t end2 = clock();
	// double elapsed_millis = double(end2 - begin2) / CLOCKS_PER_SEC;
	// cout << "BLAS DOT: " << elapsed_millis << "sec(s)" << endl;

	cout << endl;
	cout << "[pairwise_distance test]" << endl;
	x.size(3000, 256);
	y.size(4000, 256);
	beg = chrono::high_resolution_clock::now();
	blas_pairwise_distance(x, y);
	end = chrono::high_resolution_clock::now();
	//auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
	dur = end - beg;
  cout << "BLAS PDIST elapsed: " << dur.count() << " seconds" << std::endl;
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
	initialize_cuda(dev_id);
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
	initialize_cuda(dev_id);
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

void test_cluster()
{
	load_data("test.bin", "test.txt");
}

int main()
{
	// test_cuda();
	test_cuda_pdist();
}
