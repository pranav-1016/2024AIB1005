// This will automatically cover:
//  reduction

#include<omp.h>
#include<bits/stdc++.h>

using namespace std;


void reduction_odd_even(vector<int> &arr) {
    int odd = 0, even = 0;
    #pragma omp parallel for reduction(+:odd, even)
    for (int i=0; i < arr.size(); i++) {
        if (arr[i] % 2 == 0) {
            even++;
        } else {
            odd++;
        }
    }
    cout << odd << " : odd - even : " << even << endl;

}

void parallel_odd_even(vector<int> &arr) {
    int odd = 0, even = 0;
    #pragma omp parallel for shared(odd, even)
    for (int i=0; i < arr.size(); i++) {

        if (arr[i] % 2 != 0) {
            #pragma omp atomic update
            odd++;
        } else {
            #pragma omp atomic update
            even++;
        }
    }

    cout << odd << " : odd - even : " << even << endl;
}

void parallel_histogram(vector<int> &arr) {
    int no_threads = omp_get_num_threads();

    vector<vector<int>> partial(no_threads, vector<int> (10, 0));
}

int main() {
    // vector<int> arr = {1,2,3,4,5,6,7,8,9,10};
    vector<int> arr = {1,2,4,5,10,7,1,3,5,7,3,2,4,5,10,7,4,7,8,9,10,5,7,10,9};
    omp_set_num_threads(8);
    double start = omp_get_wtime();
    // parallel_odd_even(arr);
    reduction_odd_even(arr);
    double end = omp_get_wtime() - start;

    cout << " end " << end << endl;
    return 0;
}