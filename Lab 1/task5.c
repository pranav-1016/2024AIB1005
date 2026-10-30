
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void serial(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++)
                sum += A[i*n+k] * B[k*n+j];
            C[i*n+j] = sum;
        }
}

void outer(double *A, double *B, double *C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++)
                sum += A[i*n+k] * B[k*n+j];
            C[i*n+j] = sum;
        }
}

void collapse(double *A, double *B, double *C, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++)
                sum += A[i*n+k] * B[k*n+j];
            C[i*n+j] = sum;
        }
}

int check(double *A, double *B, size_t size) {
    for (size_t i = 0; i < size; i++)
        if (fabs(A[i] - B[i]) > 1e-9)
            return 0;
    return 1;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 256;
    if (n <= 0 || (size_t)n > (size_t)-1 / (size_t)n / sizeof(double))
        return 1;

    size_t size = (size_t)n * n;
    double *A = malloc(size * sizeof(double));
    double *B = malloc(size * sizeof(double));
    double *S = malloc(size * sizeof(double));
    double *P = malloc(size * sizeof(double));
    double *Q = malloc(size * sizeof(double));

    if (!A || !B || !S || !P || !Q) {
        fprintf(stderr, "Allocation failed\n");
        free(A); free(B); free(S); free(P); free(Q);
        return 1;
    }

    for (size_t i = 0; i < size; i++) {
        A[i] = (i % 17 - 8) / 10.0;
        B[i] = ((int)(i % 13) - 6) / 10.0;
    }

    double t = omp_get_wtime();
    serial(A, B, S, n);
    double st = omp_get_wtime() - t;
    printf("N=%d, Serial=%.6f s\n", n, st);

    puts("Threads | Method   | Time(s)  | Speedup | Efficiency | Check");

    for (int p = 1; p <= 8; p *= 2) {
        omp_set_num_threads(p);

        for (int method = 0; method < 2; method++) {
            double *C = method == 0 ? P : Q;
            for (size_t i = 0; i < size; i++) C[i] = 0;

            t = omp_get_wtime();
            if (method == 0) outer(A, B, C, n);
            else collapse(A, B, C, n);
            double elapsed = omp_get_wtime() - t;
            double speedup = st / elapsed;

            printf("%7d | %-8s | %.6f | %7.3f | %10.3f | %s\n",
                   p, method == 0 ? "Outer" : "Collapse",
                   elapsed, speedup, speedup / p,
                   check(S, C, size) ? "PASS" : "FAIL");
        }
    }

    free(A); free(B); free(S); free(P); free(Q);
    return 0;
}
