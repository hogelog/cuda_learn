#include <stdio.h>
#include "sharedmem.cuh"

__global__ void cuda002Kernel( float* g_idata, float* g_odata) 
{
    SharedMemory<float> smem;
    float* sdata = smem.getPointer();

    // スレッドIDを取得
    const unsigned int tid = threadIdx.x;

    //グローバルメモリから入力データの読み込み
    sdata[tid] = g_idata[tid];
    __syncthreads();

    //ここで計算を行う
    sdata[tid] = (float) 2 * sdata[tid];
    __syncthreads();

    //グローバルメモリに結果を書き込む
    g_odata[tid] = sdata[tid];
}
