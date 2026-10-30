#include <omp.h>
#include <iostream>
#include <vector>
#include <iomanip>
using namespace std;

int serial_search(vector<int>& a, int n, int target)
{
    for (int i = 0; i < n; i++) {
        if (a[i] == target)
            return i;
    }
    return -1;
}

int parallel_search(vector<int>& a, int n, int target)
{
    int index = n;

    #pragma omp parallel for reduction(min:index)
    for (int i = 0; i < n; i++) {
        if (a[i] == target && i < index)
            index = i;
    }

    if (index == n)
        return -1;

    return index;
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        cout << "Usage: " << argv[0]
             << " <array_size> <target>\n";
        return 1;
    }

    int n = atoi(argv[1]);
    int target = atoi(argv[2]);

    if (n <= 0) {
        cout << "Array size must be positive.\n";
        return 1;
    }

    vector<int> a(n);

    // Initialize array
    for (int i = 0; i < n; i++)
        a[i] = i % 1000;

    // Insert target multiple times for testing
    // if (n > 10) {
    //     a[10] = target;
    //     a[n - 1] = target;
    // } else {
    //     a[n - 1] = target;
    // }

    // Serial search
    double start = omp_get_wtime();
    int serial_index = serial_search(a, n, target);
    double serial_time = omp_get_wtime() - start;

    cout << "Array size: " << n << "\n";
    cout << "Target: " << target << "\n";
    cout << fixed << setprecision(6);
    cout << "Serial index: " << serial_index << "\n";
    cout << "Serial time: " << serial_time << " seconds\n\n";

    cout << "Threads\tParallel Time\tSpeedup\t\tCorrect\n";

    int thread_counts[] = {1, 2, 4, 8, 16};

    for (int t = 0; t < 5; t++) {
        int threads = thread_counts[t];
        omp_set_num_threads(threads);

        double times[3];
        int parallel_index = -1;

        // Take three measurements
        for (int r = 0; r < 3; r++) {
            start = omp_get_wtime();
            parallel_index = parallel_search(a, n, target);
            times[r] = omp_get_wtime() - start;
        }

        // Sort three times to find the median
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
        bool correct = (serial_index == parallel_index);

        double speedup = 0;
        if (parallel_time > 0)
            speedup = serial_time / parallel_time;
        cout << fixed << setprecision(6);
        cout << threads << "\t"
             << parallel_time << "\t"
             << speedup << "\t\t"
             << (correct ? "PASS" : "FAIL") << "\n";

        if (!correct) {
            cout << "Results do not match!\n";
            return 1;
        }
    }

    return 0;
}
