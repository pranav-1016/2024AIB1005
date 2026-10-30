
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include <omp.h>

void serial(int *a, long n, int *mx, long *mi, int *mn, long *ni) {
    *mx = *mn = a[0]; *mi = *ni = 0;
    for (long i = 1; i < n; i++) {
        if (a[i] > *mx) *mx = a[i], *mi = i;
        if (a[i] < *mn) *mn = a[i], *ni = i;
    }
}

void reduction(int *a, long n, int *mx, int *mn) {
    int x = INT_MIN, y = INT_MAX;
    #pragma omp parallel for reduction(max:x) reduction(min:y)
    for (long i = 0; i < n; i++) {
        if (a[i] > x) x = a[i];
        if (a[i] < y) y = a[i];
    }
    *mx = x; *mn = y;
}

void critical_idx(int *a, long n, int *v, long *idx, int ismax) {
    *v = a[0]; *idx = 0;
    #pragma omp parallel for
    for (long i = 1; i < n; i++) {
        #pragma omp critical
        {
            if (ismax ? (a[i] > *v || (a[i] == *v && i < *idx)) : (a[i] < *v || (a[i] == *v && i < *idx))) {
                *v = a[i];
                *idx = i;
            }
        }
    }
}

void manual_idx(int *a, long n, int *v, long *idx, int ismax) {
    int nt = omp_get_max_threads();
    int *vals = malloc(nt * sizeof(int));
    long *ids = malloc(nt * sizeof(long));

    #pragma omp parallel
    {
        int t = omp_get_thread_num(), p = omp_get_num_threads();
        long lo = n * t / p, hi = n * (t + 1) / p;
        int best = ismax ? INT_MIN : INT_MAX;
        long bi = lo;
        for (long i = lo; i < hi; i++)
            if (ismax ? a[i] > best : a[i] < best)
                best = a[i], bi = i;
        vals[t] = best; ids[t] = bi;
    }

    *v = vals[0]; *idx = ids[0];
    for (int t = 1; t < nt; t++)
        if (ismax ? vals[t] > *v : vals[t] < *v)
            *v = vals[t], *idx = ids[t];

    free(vals); free(ids);
}

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 10000000;
    if (n <= 0 || (unsigned long)n > SIZE_MAX / sizeof(int)) return 1;
    int *a = malloc((size_t)n * sizeof(int));
    if (!a) return 1;

    for (long i = 0; i < n; i++)
        a[i] = (int)(((i * 48271LL + 17) % 2000001LL) - 1000000LL);

    int mx, mn, x, y; long mi, ni, idx1, idx2;
    double t = omp_get_wtime();
    serial(a, n, &mx, &mi, &mn, &ni);
    double st = omp_get_wtime() - t;
    printf("N=%ld Serial: max=%d @%ld, min=%d @%ld, time=%.6f s\n",
           n, mx, mi, mn, ni, st);

    t = omp_get_wtime(); reduction(a, n, &x, &y);
    printf("Reduction: max=%d, min=%d, time=%.6f s, %s\n",
           x, y, omp_get_wtime()-t, x==mx && y==mn ? "PASS":"FAIL");

    puts("Threads | Critical(s) | Manual(s) | Critical | Manual");
    for (int p = 1; p <= 8; p *= 2) {
        omp_set_num_threads(p);
        t = omp_get_wtime();
        critical_idx(a,n,&x,&idx1,1); critical_idx(a,n,&y,&idx2,0);
        double ct = omp_get_wtime()-t;
        int cok = x==mx && idx1==mi && y==mn && idx2==ni;

        t = omp_get_wtime();
        manual_idx(a,n,&x,&idx1,1); manual_idx(a,n,&y,&idx2,0);
        double mt = omp_get_wtime()-t;
        int mok = x==mx && idx1==mi && y==mn && idx2==ni;
        printf("%7d | %11.6f | %9.6f | %8s | %6s\n",
               p, ct, mt, cok?"PASS":"FAIL", mok?"PASS":"FAIL");
    }
    free(a);
    return 0;
}
