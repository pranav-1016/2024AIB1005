#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

void vec_add_serial(double *a, double *b, double *c, long n) {
    for (long i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

void vec_add_parallel(double *a, double *b, double *c, long n) {
    #pragma omp parallel for
    for (long i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    char *endptr;
    long n = strtol(argv[1], &endptr, 10);

    if (*endptr != '\0' || n <= 0) {
        fprintf(stderr, "Array size must be a positive integer.\n");
        return 1;
    }

    if ((unsigned long)n > (unsigned long)(-1) /
                              (4 * sizeof(double))) {
        fprintf(stderr, "Array size is too large.\n");
        return 1;
    }

    double *a = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *c_serial = malloc((size_t)n * sizeof(double));
    double *c_parallel = malloc((size_t)n * sizeof(double));

    if (a == NULL || b == NULL ||
        c_serial == NULL || c_parallel == NULL) {
        fprintf(stderr, "Memory allocation failed for N = %ld.\n", n);
        free(a);
        free(b);
        free(c_serial);
        free(c_parallel);
        return 1;
    }

    // Small integer-valued doubles make addition exact.
    for (long i = 0; i < n; i++) {
        a[i] = (double)(i % 1000);
        b[i] = (double)(i % 100);
    }

    // Serial baseline
    double start = omp_get_wtime();
    vec_add_serial(a, b, c_serial, n);
    double serial_time = omp_get_wtime() - start;

    printf("Array size: %ld\n", n);
    printf("Serial time: %.6f seconds\n\n", serial_time);
    printf("Threads\tParallel time\tSpeedup\t\tCorrect\n");

    // Use explicit thread counts, independent of OMP_NUM_THREADS.
    int thread_counts[] = {1, 2, 4, 8, 16};
    int count = (int)(sizeof(thread_counts) /
                      sizeof(thread_counts[0]));

    for (int t = 0; t < count; t++) {
        int threads = thread_counts[t];

        omp_set_num_threads(threads);

        double parallel_time = 0.0;

        // Take 3 measurements and use the median.
        double times[3];

        for (int r = 0; r < 3; r++) {
            start = omp_get_wtime();
            vec_add_parallel(a, b, c_parallel, n);
            times[r] = omp_get_wtime() - start;
        }

        // Sort 3 times to get the median.
        if (times[0] > times[1]) {
            double tmp = times[0];
            times[0] = times[1];
            times[1] = tmp;
        }
        if (times[1] > times[2]) {
            double tmp = times[1];
            times[1] = times[2];
            times[2] = tmp;
        }
        if (times[0] > times[1]) {
            double tmp = times[0];
            times[0] = times[1];
            times[1] = tmp;
        }

        parallel_time = times[1];

        int correct = 1;
        for (long i = 0; i < n; i++) {
            if (c_serial[i] != c_parallel[i]) {
                correct = 0;
                fprintf(stderr,
                        "Mismatch at index %ld: %.17g != %.17g\n",
                        i, c_serial[i], c_parallel[i]);
                break;
            }
        }

        printf("%d\t%.6f\t\t%.3f\t\t%s\n",
               threads, parallel_time,
               serial_time / parallel_time,
               correct ? "PASS" : "FAIL");

        if (!correct) {
            free(a);
            free(b);
            free(c_serial);
            free(c_parallel);
            return 1;
        }
    }

    free(a);
    free(b);
    free(c_serial);
    free(c_parallel);

    return 0;
}