#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>

using namespace std;

void count_serial(const vector<int>& a, long long& even_count, long long& odd_count) {
    even_count = 0;
    odd_count = 0;

    for (int x : a) {
        if (x % 2 == 0) even_count++;
        else odd_count++;
    }
}

void count_reduction(const vector<int>& a, int threads, long long& even_count, long long& odd_count) {
    long long even = 0, odd = 0;
    long long n = static_cast<long long>(a.size());

    #pragma omp parallel for num_threads(threads) reduction(+:even, odd)
    for (long long i = 0; i < n; i++) {
        if (a[i] % 2 == 0) even++;
        else odd++;
    }

    even_count = even;
    odd_count = odd;
}

void count_atomic(const vector<int>& a, int threads, long long& even_count, long long& odd_count) {
    long long even = 0, odd = 0;
    long long n = static_cast<long long>(a.size());

    #pragma omp parallel for num_threads(threads) shared(even, odd)
    for (long long i = 0; i < n; i++) {
        if (a[i] % 2 == 0) {
            #pragma omp atomic update
            even++;
        } else {
            #pragma omp atomic update
            odd++;
        }
    }

    even_count = even;
    odd_count = odd;
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
    const vector<int> sizes = {100000, 1000000, 10000000, 100000000};
    const vector<int> thread_counts = {1, 2, 4, 8};

    for (int n : sizes) {
        cout << "\n===== Even/odd counting: N = " << n << " =====\n";
        vector<int> a(n);
        for (int i = 0; i < n; i++) a[i] = i;

        long long serial_even, serial_odd;
        double start = omp_get_wtime();
        count_serial(a, serial_even, serial_odd);
        double serial_time = omp_get_wtime() - start;

        cout << "Serial counts: even = " << serial_even
             << ", odd = " << serial_odd << '\n';
        cout << left << setw(18) << "Method" << right
             << setw(8) << "Threads" << setw(14) << "Time(s)"
             << setw(12) << "Speedup" << setw(13) << "Efficiency\n";
        cout << string(65, '-') << '\n';
        report("Serial", 1, serial_time, serial_time);

        for (int threads : thread_counts) {
            long long even, odd;

            start = omp_get_wtime();
            count_reduction(a, threads, even, odd);
            double elapsed = omp_get_wtime() - start;
            bool correct = even == serial_even && odd == serial_odd && even + odd == n;
            cout << "Reduction correctness: " << (correct ? "PASS" : "FAIL") << '\n';
            report("Reduction", threads, elapsed, serial_time);

            start = omp_get_wtime();
            count_atomic(a, threads, even, odd);
            elapsed = omp_get_wtime() - start;
            correct = even == serial_even && odd == serial_odd && even + odd == n;
            cout << "Atomic correctness: " << (correct ? "PASS" : "FAIL") << '\n';
            report("Atomic", threads, elapsed, serial_time);
        }
    }
    return 0;
}
