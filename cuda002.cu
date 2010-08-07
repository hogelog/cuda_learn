#include <stdio.h>
#include <cutil.h>

#include "cuda002_kernel.cu"
int main( int argc, char** argv) 
{
    //デバイスの初期化
    CUT_DEVICE_INIT(argc, argv);

    //タイマーを作成して計測開始
    unsigned int timer = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));

    //メインメモリ上にfloat型のデータを100個生成する
    float* h_idata = (float*) malloc(sizeof( float) * 100);
    for( int i = 0; i < 100; i++) 
    {
        h_idata[i] = i;
    }

    //デバイス上（ビデオカードのこと）にも同じくfloat型100個分のメモリを確保する
    float* d_idata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, sizeof( float) * 100 ));
    //メインメモリからデバイスのメモリにデータを転送する
    CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, sizeof( float) * 100 , cudaMemcpyHostToDevice) );

    //デバイス上に結果を格納するfloat型100個分のメモリを確保する
    float* d_odata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, sizeof( float) * 100));

    dim3  grid( 1, 1, 1);
    //100は100個並列であるため
    dim3  threads(100, 1, 1);

    //ここでGPUを使った計算が行われる
    cuda002Kernel<<< grid, threads, sizeof( float) * 100 >>>( d_idata, d_odata);
        
    // 結果をコピーするための領域を確保する
    float* h_odata = (float*) malloc(sizeof( float) * 100);
    //デバイスからメインメモリ上に実行結果をコピー
    CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_odata, sizeof( float) * 100, cudaMemcpyDeviceToHost) );

    //実行結果を表示
    printf("入力データ , 出力データ\n");
    for (int i=0;i<100;i++)
    {
        printf("%f , %f\n",h_idata[i],h_odata[i]);
    }
    //タイマーを停止しかかった時間を表示
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    CUT_SAFE_CALL( cutDeleteTimer( timer));

    //各種メモリを解放
    free( h_idata);
    free( h_odata);
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_odata));

    //終了処理
    CUT_EXIT(argc, argv);
}
