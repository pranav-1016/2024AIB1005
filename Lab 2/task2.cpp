#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
using namespace std;

void serial_sort(vector<int>& a, int n) {
    // Odd-even transposition sort
    for (int phase = 0; phase < n; phase++) {

        // Even phase
        for (int i = 0; i < n - 1; i += 2) {
            if (a[i] > a[i + 1]) {
                int temp = a[i];
                a[i] = a[i + 1];
                a[i + 1] = temp;
            }
        }

        // Odd phase
        for (int i = 1; i < n - 1; i += 2) {
            if (a[i] > a[i + 1]) {
                int temp = a[i];
                a[i] = a[i + 1];
                a[i + 1] = temp;
            }
        }
    }
}

void parallel_sort(vector<int>& a, int n) {
    for (int phase = 0; phase < n; phase++) {

        // Even phase
        #pragma omp parallel for
        for (int i = 0; i < n - 1; i += 2) {
            if (a[i] > a[i + 1]) {
                int temp = a[i];
                a[i] = a[i + 1];
                a[i + 1] = temp;
            }
        }

        // Odd phase
        #pragma omp parallel for
        for (int i = 1; i < n - 1; i += 2) {
            if (a[i] > a[i + 1]) {
                int temp = a[i];
                a[i] = a[i + 1];
                a[i + 1] = temp;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <array_size>\n";
        return 1;
    }

    int n = atoi(argv[1]);

    if (n <= 0) {
        cout << "Array size must be positive.\n";
        return 1;
    }

    vector<int> original(n);
    vector<int> serial(n);
    vector<int> parallel(n);

    // Generate reproducible input
    for (int i = 0; i < n; i++)
        original[i] = (i * 17 + 23) % 1000;

    serial = original;
    parallel = original;

    // Serial baseline
    double start = omp_get_wtime();
    serial_sort(serial, n);
    double serial_time = omp_get_wtime() - start;

    cout << fixed << setprecision(6);
    cout << "Array size: " << n << "\n";
    cout << "Serial time: " << serial_time << " seconds\n\n";
    cout << "Threads\tParallel Time\tSpeedup\t\tCorrect\n";

    int thread_counts[] = {1, 2, 4, 8, 16};

    for (int t = 0; t < 5; t++) {
        int threads = thread_counts[t];
        omp_set_num_threads(threads);

        double times[3];

        for (int r = 0; r < 3; r++) {
            // Restore original input before every measurement
            parallel = original;

            start = omp_get_wtime();
            parallel_sort(parallel, n);
            times[r] = omp_get_wtime() - start;
        }

        // Sort the three times to obtain the median
        if (times[0] > times[1]) {
            double temp = times[0];
            times[0] = times[1];
            times[1] = temp;
        }

        if (times[1] > times[2]) {
            double temp = times[1];
            times[1] = times[2];
            times[2] = temp;
        }

        if (times[0] > times[1]) {
            double temp = times[0];
            times[0] = times[1];
            times[1] = temp;
        }

        double parallel_time = times[1];

        // Verify sorted output against serial output
        bool correct = true;

        for (int i = 0; i < n; i++) {
            if (serial[i] != parallel[i]) {
                correct = false;
                break;
            }
        }

        // Also verify that the output is sorted
        for (int i = 1; i < n; i++) {
            if (parallel[i - 1] > parallel[i]) {
                correct = false;
                break;
            }
        }

        double speedup = 0.0;
        if (parallel_time > 0.0)
            speedup = serial_time / parallel_time;

        cout << threads << "\t"
             << parallel_time << "\t"
             << speedup << "\t\t"
             << (correct ? "PASS" : "FAIL") << "\n";

        if (!correct) {
            cout << "Sorting verification failed!\n";
            return 1;
        }
    }

    return 0;
}