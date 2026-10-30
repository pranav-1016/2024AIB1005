#include "omp.h"
#include<stdio.h>
int main() {
    // int tsum = 0;
    // #pragma omp parallel
    // {
    //     tsum += 1;
    // }
    // printf("Sum is %d\n", tsum);
    int arr[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17};
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i=0; i < 17; i++) {
        // printf("hello world %d\n", omp_get_thread_num());
        sum += arr[i];
    }
    printf("Sum : %d\n", sum);
    return 0;
}