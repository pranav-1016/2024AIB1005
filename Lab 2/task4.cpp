
#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <cstdlib>
using namespace std;

void serial_transpose(vector<vector<int>>& A,
                      vector<vector<int>>& B, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B[j][i] = A[i][j];
        }
    }
}

void parallel_transpose(vector<vector<int>>& A,
                        vector<vector<int>>& B, int m, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B[j][i] = A[i][j];
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
        cout << "Usage: ./task4 rows columns\n";
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);

    if (m <= 0 || n <= 0) {
        cout << "Rows and columns must be positive.\n";
        return 1;
    }

    vector<vector<int>> A(m, vector<int>(n));
    vector<vector<int>> serial_B(n, vector<int>(m));
    vector<vector<int>> parallel_B(n, vector<int>(m));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (i * 17 + j * 13) % 1000;
        }
    }

    // Serial baseline
    double start = omp_get_wtime();
    serial_transpose(A, serial_B, m, n);
    double serial_time = omp_get_wtime() - start;

    cout << fixed << setprecision(6);
    cout << "Matrix size: " << m << " x " << n << "\n";
    cout << "Serial time: " << serial_time << " seconds\n\n";

    int threads[] = {1, 2, 4, 8, 16};

    cout << "Threads\tTime(s)\t\tSpeedup\t\tCorrectness\n";

    for (int t : threads) {
        omp_set_num_threads(t);

        double times[3];

        for (int r = 0; r < 3; r++) {
            start = omp_get_wtime();
            parallel_transpose(A, parallel_B, m, n);
            times[r] = omp_get_wtime() - start;
        }

        double parallel_time = median3(times[0], times[1], times[2]);

        bool correct = true;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (parallel_B[i][j] != serial_B[i][j]) {
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
