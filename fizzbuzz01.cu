#include <stdio.h>
#include <assert.h>
#define COUNT 40

enum FizzBuzzEnum {
    FIZZBUZZ, FIZZ, BUZZ, NONE
};

__device__ FizzBuzzEnum to_enum(int num) {
    return (num%15)==0 ? FIZZBUZZ : (num%3)==0 ? FIZZ : (num%5)==0 ? BUZZ : NONE;
}
__global__ void dev_fizzbuzz(FizzBuzzEnum *d_fizzbuzz) {
    int i = threadIdx.x;
    d_fizzbuzz[i] = to_enum(i+1);
}

int main() {
    FizzBuzzEnum h_fizzbuzz[COUNT];
    FizzBuzzEnum *d_fizzbuzz;

    cudaMalloc(&d_fizzbuzz, sizeof(h_fizzbuzz));

    dev_fizzbuzz<<<1,COUNT>>>(d_fizzbuzz);

    cudaMemcpy(h_fizzbuzz, d_fizzbuzz, sizeof(h_fizzbuzz), cudaMemcpyDeviceToHost);

    for (int i=0;i<COUNT;++i) {
	switch(h_fizzbuzz[i]) {
	    case FIZZBUZZ:
		puts("FizzBuzz");
		break;
	    case FIZZ:
		puts("Fizz");
		break;
	    case BUZZ:
		puts("Buzz");
		break;
	    case NONE:
		printf("%d\n", i+1);
		break;
	    default:
		assert(0);
	}
    }

    cudaFree(d_fizzbuzz);

    return 0;
}
// vim: set ft=cpp:
