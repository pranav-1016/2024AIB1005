#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <omp.h>

using namespace std;

const int CUTOFF = 2048;

void merge(vector<int>& a, vector<int>& tmp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right) {
        if (a[i] <= a[j]) tmp[k++] = a[i++];
        else tmp[k++] = a[j++];
    }
    while (i <= mid) tmp[k++] = a[i++];
    while (j <= right) tmp[k++] = a[j++];

    for (int p = left; p <= right; p++) a[p] = tmp[p];
}

void merge_sort_serial(vector<int>& a, vector<int>& tmp, int left, int right) {
    if (left >= right) return;

    int mid = left + (right - left) / 2;
    merge_sort_serial(a, tmp, left, mid);
    merge_sort_serial(a, tmp, mid + 1, right);
    merge(a, tmp, left, mid, right);
}

void merge_sort_parallel(vector<int>& a, vector<int>& tmp, int left, int right) {
    if (left >= right) return;

    if (right - left + 1 <= CUTOFF) {
        merge_sort_serial(a, tmp, left, right);
        return;
    }

    int mid = left + (right - left) / 2;

    #pragma omp task shared(a, tmp) firstprivate(left, mid)
    merge_sort_parallel(a, tmp, left, mid);

    #pragma omp task shared(a, tmp) firstprivate(mid, right)
    merge_sort_parallel(a, tmp, mid + 1, right);

    #pragma omp taskwait
    merge(a, tmp, left, mid, right);
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
        cout << "\n===== Merge sort: N = " << n << " =====\n";
        vector<int> original(n);
        unsigned int seed = 12345;
        for (int i = 0; i < n; i++) {
            seed = seed * 1664525u + 1013904223u;
            original[i] = static_cast<int>(seed % 1000000);
        }

        vector<int> serial = original, tmp(n);
        double start = omp_get_wtime();
        merge_sort_serial(serial, tmp, 0, n - 1);
        double serial_time = omp_get_wtime() - start;

        cout << left << setw(18) << "Method" << right
             << setw(8) << "Threads" << setw(14) << "Time(s)"
             << setw(12) << "Speedup" << setw(13) << "Efficiency\n";
        cout << string(65, '-') << '\n';
        report("Serial", 1, serial_time, serial_time);
        cout << "Serial sorted: " << (is_sorted(serial.begin(), serial.end()) ? "PASS" : "FAIL") << '\n';

        for (int threads : thread_counts) {
            vector<int> parallel = original, parallel_tmp(n);
            start = omp_get_wtime();

            #pragma omp parallel num_threads(threads)
            {
                #pragma omp single
                merge_sort_parallel(parallel, parallel_tmp, 0, n - 1);
            }

            double elapsed = omp_get_wtime() - start;
            bool correct = parallel == serial && is_sorted(parallel.begin(), parallel.end());
            cout << "Parallel correctness: " << (correct ? "PASS" : "FAIL") << '\n';
            report("OpenMP tasks", threads, elapsed, serial_time);
        }
    }
    return 0;
}
