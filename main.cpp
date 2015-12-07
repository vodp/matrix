#include "matrix.hpp"
#include "matrixview.hpp"
#include "alg.hpp"
#include <functional>
#include <cctype>
#include <locale>
#include <chrono>

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

matrix<int> load_labels (const char *labelfile) {
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
	matrix<int> labels(total_points, 1);
	for (int i=0; i < clusters.size(); ++i)
		for (int j=0; j < clusters[i].size(); ++j)
			labels(clusters[i][j], 0) = i;
	return labels;
}

matrix<float> compute_cluster_centers (matrix<float>& X, matrix<int>& y) {
	assert(X.rows() == y.size());
	int num_classes = y.max() + 1;
	matrix<float> centers(num_classes, X.cols());
	size_t m = X.rows();
	for (int i=0; i < num_classes; ++i) {
		centers.r_(i) = X.r_(y == i).mean(0);
	}
	return centers;
}
matrix<float>& compute_coefficients (matrix<float>& X, matrix<int>& y, matrix<float>& centers) {
	int num_clusters = y.max() + 1;
	matrix<float> avg_coeffs(1, num_clusters);
	for (int i=0; i < num_clusters; ++i) {
		cout << "Computing cluster #" << i << "..." << endl;
		matrix<float> thiscluster = X.r_(y == i).detach();
		matrix<float> inter_dist = cuda_pairwise_distance (thiscluster, centers);
		matrix<size_t> sorted_ix = matrix_argsort(inter_dist, matrix<size_t>::SORT_ROWS);
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
		matrix<float> intra_dist = cuda_pairwise_distance (thiscluster);
		matrix<float> coeffs(1, nearest_clusters.size());

		for (size_t k=0; k < unique_nearest_clusters.size(); ++k) {
			size_t next_cluster = unique_nearest_clusters[k];
			matrix<float> subcluster = X.r_(y == (int)next_cluster).detach();

			matrix<unsigned char> bool_ix = nearest_clusters == next_cluster;
			matrix<float> vecs = thiscluster.r_(bool_ix).detach();
			matrix<float> mean_b = cuda_pairwise_distance(vecs, subcluster).mean(1);
			matrix<float> mean_a = intra_dist.r_(bool_ix).mean(1);
			assert(mean_a.size() == mean_b.size() && vecs.rows() == mean_a.size());
			for (int j=0; j < mean_a.rows(); ++j) {
				float coeff = (mean_b(j,0) - mean_a(j,0))/std::max(mean_b(j,0), mean_a(j,0));
				coeffs(0, count++) = coeff;
			}
		}
		assert(count == coeffs.size());
		avg_coeffs(0, i) = coeffs.mean();
		cout << "coefficient #" << i << ": " << avg_coeffs(0,i) << endl;
	}
	return avg_coeffs;
}
void csil () {
	cout << "Loading features..." << endl;
	matrix<float> X = matrix<float>::load("../dat/20m_signatures_random.caffe.256", 18389592, 256);
	X = X.c_(0, 16).detach();
	cout << "...has dimension (" << X.rows() << ", " << X.cols() << ")" << endl;
	cout << "Loading labels..." << endl;
	matrix<int> y = load_labels("../dat/cluster_20msig_5kcenter_random.lst");
	cout << "...has dimension (" << y.rows() << ", " << y.cols() << ")" << endl;
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
	matrix<float> coeffs = compute_coefficients (X, y, centers);
	matrix<float>::dump("../dat/coeffs.bin", coeffs);
	cout << "Dumped results to ../dat/coeffs.bin. DONE.";
}

// trim from start
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
