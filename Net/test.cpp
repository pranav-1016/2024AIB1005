#include "omp.h"
#include<stdio.h>
int main() {
    printf("Hello from main thread\n");
    printf("%d \n" ,omp_get_num_threads());
    printf("%d \n" ,omp_get_thread_num());
    printf("%d \n" ,omp_get_num_procs());

    #pragma omp parallel
    {
        printf("During fork %d \n" ,omp_get_thread_num());
        printf("Hello World\n");
    }

    return 0;
}