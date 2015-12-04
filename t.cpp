#include "m.hpp"

matrix<float> f() {
    matrix<float> a(1,2);
    return a;
}
matrix<float> g() {
    matrix<float> *b = new matrix<float>(1,2);
    return *b;
}
int main() {
    matrix<float> a;
    a = f(); // (*)
    cout << "--" << endl;
    a = g(); // (**)
    cout << "--" << endl;
    a = a.mean(1);  // (***)
}
