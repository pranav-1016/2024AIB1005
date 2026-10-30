
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void serial(int *A, int *B, int *C, long N) {
    for (long i = 0; i < N; i++)
        for (long j = 0; j < N; j++)
            C[i*N+j] = A[i*N+j] + B[i*N+j];
}

void outer_parallel(int *A, int *B, int *C, long N) {
    #pragma omp parallel for
    for (long i = 0; i < N; i++)
        for (long j = 0; j < N; j++)
            C[i*N+j] = A[i*N+j] + B[i*N+j];
}

void collapse_parallel(int *A, int *B, int *C, long N) {
    #pragma omp parallel for collapse(2)
    for (long i = 0; i < N; i++)
        for (long j = 0; j < N; j++)
            C[i*N+j] = A[i*N+j] + B[i*N+j];
}

double benchmark(void (*fn)(int*,int*,int*,long), int *A, int *B, int *C, long N, int reps) {
    double start = omp_get_wtime();
    for (int r = 0; r < reps; r++)
        fn(A, B, C, N);
    return (omp_get_wtime() - start) / reps;
}

int main(int argc, char **argv) {
    long N = argc > 1 ? atol(argv[1]) : 8;
    if (N <= 0 || (size_t)N > (size_t)-1 / (size_t)N / sizeof(int))
        return 1;

    size_t size = (size_t)N * N;
    int *A = malloc(size * sizeof(int));
    int *B = malloc(size * sizeof(int));
    int *S = malloc(size * sizeof(int));
    int *P = malloc(size * sizeof(int));
    int *C = malloc(size * sizeof(int));

    if (!A || !B || !S || !P || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        free(A); free(B); free(S); free(P); free(C);
        return 1;
    }

    for (size_t i = 0; i < size; i++) {
        A[i] = i % 100;
        B[i] = (i * 3) % 100;
    }

    double t = omp_get_wtime();
    serial(A, B, S, N);
    double st = omp_get_wtime() - t;
    printf("N=%ld, Serial=%.6f s\n", N, st);

    int reps = N <= 8 ? 10000 : 3;
    puts("Threads | Outer(s) | Collapse(s) | Outer | Collapse");

    for (int p = 1; p <= 8; p *= 2) {
        omp_set_num_threads(p);

        double ot = benchmark(outer_parallel, A, B, P, N, reps);
        double ct = benchmark(collapse_parallel, A, B, C, N, reps);

        int opass = 1, cpass = 1;
        for (size_t i = 0; i < size; i++) {
            if (P[i] != S[i]) opass = 0;
            if (C[i] != S[i]) cpass = 0;
        }

        printf("%7d | %8.6f | %11.6f | %5s | %8s\n",
               p, ot, ct, opass ? "PASS" : "FAIL",
               cpass ? "PASS" : "FAIL");
    }

    free(A); free(B); free(S); free(P); free(C);
    return 0;
}
