
#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <cstdlib>
#include <cmath>
using namespace std;

void serial_convolution(vector<vector<double>>& A, vector<vector<double>>& B, int m, int n) {
    for (int i = 0; i < m - 2; i++) {
        for (int j = 0; j < n - 2; j++) {
            double sum = 0;

            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    sum += A[i + x][j + y];
                }
            }

            B[i][j] = sum / 9.0;
        }
    }
}

void parallel_convolution(vector<vector<double>>& A, vector<vector<double>>& B, int m, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m - 2; i++) {
        for (int j = 0; j < n - 2; j++) {
            double sum = 0;

            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    sum += A[i + x][j + y];
                }
            }

            B[i][j] = sum / 9.0;
        }
    }
}

double median3(double a, double b, double c) {
    if (a > b) { double t = a; a = b; b = t; }
    if (b > c) { double t = b; b = c; c = t; }
    if (a > b) { double t = a; a = b; b = t; }
    return b;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: ./task5 rows columns\n";
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);

    if (m < 3 || n < 3) {
        cout << "Rows and columns must be at least 3.\n";
        return 1;
    }

    vector<vector<double>> A(m, vector<double>(n));
    vector<vector<double>> serial_B(m - 2, vector<double>(n - 2));
    vector<vector<double>> parallel_B(m - 2, vector<double>(n - 2));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (i * 17 + j * 13) % 100;
        }
    }

    // Serial baseline
    double start = omp_get_wtime();
    serial_convolution(A, serial_B, m, n);
    double serial_time = omp_get_wtime() - start;

    cout << fixed << setprecision(6);
    cout << "Input size: " << m << " x " << n << "\n";
    cout << "Output size: " << m - 2 << " x " << n - 2 << "\n";
    cout << "Serial time: " << serial_time << " seconds\n\n";

    int threads[] = {1, 2, 4, 8, 16};

    cout << "Threads\tTime(s)\t\tSpeedup\t\tCorrectness\n";

    for (int t : threads) {
        omp_set_num_threads(t);

        double times[3];

        for (int r = 0; r < 3; r++) {
            start = omp_get_wtime();
            parallel_convolution(A, parallel_B, m, n);
            times[r] = omp_get_wtime() - start;
        }

        double parallel_time = median3(times[0], times[1], times[2]);

        bool correct = true;

        for (int i = 0; i < m - 2; i++) {
            for (int j = 0; j < n - 2; j++) {
                if (fabs(parallel_B[i][j] - serial_B[i][j]) > 1e-9) {
                    correct = false;
                }
            }
        }

        cout << t << "\t" << parallel_time << "\t"
             << serial_time / parallel_time << "\t\t"
             << (correct ? "PASS" : "FAIL") << "\n";
    }

    return 0;
}
