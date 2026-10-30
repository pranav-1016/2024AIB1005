#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
/* Kept exactly as given in the lab sample. */

long sum_serial(long *a, long n) {
    long sum = 0;
    for (long i = 0; i < n; i++)
        sum += a[i];
    return sum;
}

long sum_naive_parallel_BROKEN(long *a, long n) {
    long sum = 0;
    #pragma omp parallel for
    for (long i = 0; i < n; i++)
        sum += a[i];      // <-- data race, intentionally left broken for you to observe
    return sum;
}

long sum_reduction(long *a, long n) {
    long sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; i++)
        sum += a[i];
    return sum;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    char *endptr;
    long n = strtol(argv[1], &endptr, 10);

    if (*endptr != '\0' || n <= 0 ||
        (unsigned long)n > (unsigned long)(-1) / sizeof(long)) {
        fprintf(stderr, "Invalid array size.\n");
        return 1;
    }

    long *a = malloc((size_t)n * sizeof(long));

    if (a == NULL) {
        fprintf(stderr, "Memory allocation failed for N = %ld.\n", n);
        return 1;
    }

    // Use small nonnegative values to avoid sum overflow.
    for (long i = 0; i < n; i++)
        a[i] = i % 10;

    // Serial reference result and timing.
    double start = omp_get_wtime();
    long expected = sum_serial(a, n);
    double serial_time = omp_get_wtime() - start;

    printf("Array size: %ld\n", n);
    printf("Serial sum: %ld\n", expected);
    printf("Serial time: %.6f seconds\n\n", serial_time);

    int thread_counts[] = {1, 2, 4, 8};
    int count = (int)(sizeof(thread_counts) / sizeof(thread_counts[0]));

    // printf("=== Naive parallel sum (intentionally broken) ===\n");

    // for (int t = 0; t < count; t++) {
        // int threads = thread_counts[t];
        // omp_set_num_threads(threads);

        // printf("\nThreads: %d\n", threads);

        // Run multiple times to observe inconsistent results.
        // for (int r = 0; r < 5; r++) {
        //     long result = sum_naive_parallel_BROKEN(a, n);
        //     printf("Run %d: sum = %ld %s\n",
        //            r + 1, result,
        //            result == expected ? "(matches)" : "(INCORRECT)");
        // }
    // }

    printf("\n=== Correct reduction-based sum ===\n");
    printf("Threads\tTime (s)\tSpeedup\t\tCorrect\n");

    for (int t = 0; t < count; t++) {
        int threads = thread_counts[t];
        omp_set_num_threads(threads);

        // Warm-up, excluded from timing.
        long warmup = sum_reduction(a, n);
        if (warmup != expected) {
            fprintf(stderr, "Reduction warm-up failed.\n");
            free(a);
            return 1;
        }

        double times[3];
        long result = 0;

        for (int r = 0; r < 3; r++) {
            start = omp_get_wtime();
            result = sum_reduction(a, n);
            times[r] = omp_get_wtime() - start;
        }

        // Sort the three measurements to obtain their median.
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

        int correct = (result == expected);

        printf("%d\t%.6f\t\t%.3f\t\t%s\n",
               threads, times[1],
               serial_time / times[1],
               correct ? "PASS" : "FAIL");

        if (!correct) {
            fprintf(stderr, "Reduction result does not match serial.\n");
            free(a);
            return 1;
        }
    }

    free(a);
    return 0;
}


