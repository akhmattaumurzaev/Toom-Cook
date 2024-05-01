#include <iostream>
#include <iomanip>
#include <cmath>


extern "C" {
    void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

    void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
}



double* matvec(double *A, double *x, int n, int m) {
    double *y = new double[n];
    for (int i = 0; i < n; i++) {
        y[i] = 0;
    }
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            y[i] += A[i + j * n] * x[j];
        }
    }
    return y;
}


double* Vandermonde(int k) {
    double *V = new double[k * k];
     for (int i = 0; i < k; i++) {
        V[i] = 1;
    }
    for (int j = 1; j < k; j++) {
        V[j * k] = 0;
        for (int i = 1; i < k; i++) {
            V[i + j * k] = V[i + (j - 1) * k] * ((i + 1) / 2);
            if (i % 2) {
                V[i + j * k] = -V[i + j * k];
            }
        }
    }
    return V;
}


double *inverse_Vandermonde(int k) {
    double *V = Vandermonde(k);
    int *IPIV = new int[k];
    int LWORK = k * k;
    double *WORK = new double[LWORK];
    int INFO;

    dgetrf_(&k,&k,V,&k,IPIV,&INFO);
    dgetri_(&k,V,&k,IPIV,WORK,&LWORK,&INFO);

    delete[] IPIV;
    delete[] WORK;
    return V;
}


void interpolation(double *a, double *b, double *c, int n, double *V, double *inv_V, int k) {
    double *a_values = matvec(V, a, 2 * k - 1, n);
    double *b_values = matvec(V, b, 2 * k - 1, n);
    for (int i = 0; i < 2 * k - 1; i++) {
        a_values[i] *= b_values[i];
    }
    double *temp = matvec(inv_V, a_values, 2 * k - 1, 2 * k - 1);
    for (int i = 0; i < 2 * n - 1; i++) {
        c[i] += temp[i];
    }
    delete[] a_values;
    delete[] b_values;
    delete[] temp;
}


void Toom_Cook_inner(double *a, double *b, double *c, int n, double *V, double *inv_V, int k) {
    if (n <= k) {
        interpolation(a, b, c, n, V, inv_V, k);
    } else {
        int d = (n - 1) / k + 1;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                Toom_Cook_inner(a + i * d, b + j * d, c + (i + j) * d, d, V, inv_V, k);
            }
        }
    }
}


double* Toom_Cook(double *a, double *b, int n, int k) {
    double *V = Vandermonde(2 * k - 1), *inv_V = inverse_Vandermonde(2 * k - 1);
    int d = (n - 1) / k + 1;
    d *= k;
    double *ext_a = new double[d], *ext_b = new double[d];
    double *c = new double[2 * d + k];
    for (int i = 0; i < n; i++) {
        ext_a[i] = a[i];
        ext_b[i] = b[i];
        c[i] = 0;
    }
    for (int i = n; i < d; i++) {
        ext_a[i] = 0;
        ext_b[i] = 0;
        c[i] = 0;
    }
    for (int i = d; i < 2 * d + k; i++) {
        c[i] = 0;
    }
    Toom_Cook_inner(ext_a, ext_b, c, n, V, inv_V, k);
    return c;
}


void test_func(int n, int k) {
    double *a = new double[n], *b = new double[n];
    for (int i = 0; i < n; i++) {
        a[i] = i + 1;
        b[i] = 1;
    }
    double *c = Toom_Cook(a, b, n, k);
    for (int i = 0; i < 2 * n - 1; ++i) {
		std::cout << c[i] << ' ';
	}
	std::cout << std::endl;
}


int main() { 
    int k = 2, n = 15;
    test_func(n, k);
}


