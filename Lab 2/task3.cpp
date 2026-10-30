#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
using namespace std;

void serial_histogram(vector<int>& a, int n, int hist[])
{
    for (int i = 0; i < 10; i++)
        hist[i] = 0;

    for (int i = 0; i < n; i++)
        hist[a[i]]++;
}

void parallel_histogram(vector<int>& a, int n, int hist[])
{
    int threads = omp_get_max_threads();

    // One private histogram per thread
    vector<vector<int>> local_hist(threads, vector<int>(10, 0));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp for
        for (int i = 0; i < n; i++)
            local_hist[tid][a[i]]++;
    }

    // Combine the private histograms
    for (int t = 0; t < threads; t++) {
        for (int i = 0; i < 10; i++)
            hist[i] += local_hist[t][i];
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        cout << "Usage: " << argv[0]
             << " <array_size>\n";
        return 1;
    }

    int n = atoi(argv[1]);

    if (n <= 0) {
        cout << "Array size must be positive.\n";
        return 1;
    }

    vector<int> a(n);

    // Generate reproducible values from 0 to 9
    // for (int i = 0; i < n; i++)
    //     a[i] = (i * 7 + 3) % 10;
    for (int i = 0; i < n; i++)
        a[i] = (i * i + 3) % 10;
    
    int serial_hist[10] = {0};
    int parallel_hist[10] = {0};

    // Serial baseline
    double start = omp_get_wtime();
    serial_histogram(a, n, serial_hist);
    double serial_time = omp_get_wtime() - start;

    cout << fixed << setprecision(6);
    cout << "Array size: " << n << "\n";
    cout << "Serial time: " << serial_time << " seconds\n\n";

    cout << "Serial histogram:\n";
    for (int i = 0; i < 10; i++)
        cout << i << ": " << serial_hist[i] << "\n";

    cout << "\nThreads\tParallel Time\tSpeedup\t\tCorrect\n";

    int thread_counts[] = {1, 2, 4, 8, 16};

    for (int t = 0; t < 5; t++) {
        int threads = thread_counts[t];
        omp_set_num_threads(threads);

        double times[3];

        for (int r = 0; r < 3; r++) {
            for (int i = 0; i < 10; i++)
                parallel_hist[i] = 0;

            start = omp_get_wtime();
            parallel_histogram(a, n, parallel_hist);
            times[r] = omp_get_wtime() - start;
        }

        // Sort three times to obtain the median
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

        // Verify all 10 histogram bins
        bool correct = true;

        for (int i = 0; i < 10; i++) {
            if (serial_hist[i] != parallel_hist[i]) {
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
            cout << "Histogram verification failed!\n";
            return 1;
        }
    }

    return 0;
}