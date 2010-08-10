#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CORE_NUM (32)

#define WIDTH  (256)
#define HEIGHT (256)

#define SIZE (WIDTH*HEIGHT)

#define SCALE (1.0)
#define ENDIAN (-1)

__global__ void white2black(float *ppmbuf) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    __syncthreads();
    float gray = (threadIdx.x / (float)WIDTH);
    ppmbuf[index] = gray;
}
__host__ void writeppm(FILE *output, const size_t size) {
    float *ppmbuf = (float*)malloc(sizeof(float)*size);
    float *d_ppmbuf = NULL;

    dim3 blockNum(HEIGHT);
    dim3 threadNum(WIDTH);

    cudaMalloc((void**)&d_ppmbuf, sizeof(float)*size);

    if (cudaConfigureCall(blockNum, threadNum)) {
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaGetLastError()));
        exit(EXIT_FAILURE);
    }
    white2black<<<blockNum, threadNum>>>(d_ppmbuf);

    cudaMemcpy(ppmbuf, d_ppmbuf, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaFree(d_ppmbuf);

    fprintf(output, "Pf\n%d %d\n%f\n", WIDTH, HEIGHT, ENDIAN*SCALE);
    fwrite(ppmbuf, sizeof(float), size, output);
    free(ppmbuf);
}

int main(int argc, char **argv) {
    const size_t size = WIDTH * HEIGHT;
    FILE *ppm = fopen("white2black.ppm", "w");
    if (ppm) {
        writeppm(ppm, size);
    }
    return 0;
}
