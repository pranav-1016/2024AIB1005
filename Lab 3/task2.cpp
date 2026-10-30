#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <omp.h>

using namespace std;

// The first 5% of iterations are expensive, creating clustered uneven work.
inline double expensive_operation(int i, int n, double input) {
    double x = input;
    int work = (i < n / 20) ? 100 : 2;

    for (int j = 0; j < work; j++)
        x = x * 1.0000001 + 0.0000001;

    return x;
}

void report(const char* method, int threads, double time, double serial_time) {
    double speedup = serial_time / time;
    double efficiency = speedup / threads * 100.0;
    cout << left << setw(18) << method
         << right << setw(8) << threads
         << setw(14) << fixed << setprecision(6) << time
         << setw(12) << setprecision(3) << speedup
         << setw(12) << efficiency << "%\n";
}

int main() {
    const vector<int> sizes = {100000, 1000000, 10000000};
    const vector<int> thread_counts = {1, 2, 4, 8};

    for (int n : sizes) {
        cout << "\n===== Scheduling: N = " << n << " =====\n";
        vector<double> a(n), serial(n), output(n);
        for (int i = 0; i < n; i++) a[i] = i * 0.001;

        double start = omp_get_wtime();
        for (int i = 0; i < n; i++)
            serial[i] = expensive_operation(i, n, a[i]);
        double serial_time = omp_get_wtime() - start;

        cout << left << setw(18) << "Method" << right
            << setw(8) << "Threads" << setw(14) << "Time(s)"
            << setw(12) << "Speedup" << setw(13) << "Efficiency\n";
        cout << string(65, '-') << '\n';
        report("Serial", 1, serial_time, serial_time);

        for (int threads : thread_counts) {
            start = omp_get_wtime();
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (int i = 0; i < n; i++)
                output[i] = expensive_operation(i, n, a[i]);
            double elapsed = omp_get_wtime() - start;
            cout << "Static correctness: " << (output == serial ? "PASS" : "FAIL") << '\n';
            report("Static", threads, elapsed, serial_time);

            start = omp_get_wtime();
            #pragma omp parallel for num_threads(threads) schedule(dynamic, 100)
            for (int i = 0; i < n; i++)
                output[i] = expensive_operation(i, n, a[i]);
            elapsed = omp_get_wtime() - start;
            cout << "Dynamic correctness: " << (output == serial ? "PASS" : "FAIL") << '\n';
            report("Dynamic (100)", threads, elapsed, serial_time);

            start = omp_get_wtime();
            #pragma omp parallel for num_threads(threads) schedule(guided)
            for (int i = 0; i < n; i++)
                output[i] = expensive_operation(i, n, a[i]);
            elapsed = omp_get_wtime() - start;
            cout << "Guided correctness: " << (output == serial ? "PASS" : "FAIL") << '\n';
            report("Guided", threads, elapsed, serial_time);
        }
    }
    return 0;
}
