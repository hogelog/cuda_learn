#include <stdio.h>
__device__ void dev_strcpy(char *dst, const char *src) {
    while (*dst++ = *src++);
}
__global__ void gen_hello(char *buf) {
    dev_strcpy(buf, "Hello, World!");
}
int main(){
    char *hello_ptr;
    char hello_buf[128];

    cudaMalloc(&hello_ptr, 128);

    gen_hello<<<1,1>>>(hello_ptr);

    cudaMemcpy(hello_buf, hello_ptr, 128, cudaMemcpyDeviceToHost);

    cudaFree(hello_ptr);

    puts(hello_buf);

    return 0;
}
// vim: set ft=cpp:
