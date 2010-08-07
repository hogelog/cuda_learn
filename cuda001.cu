#include <stdio.h>
#include <cutil.h>
int main( int argc, char** argv) 
{
    CUT_DEVICE_INIT(argc, argv);

    unsigned int timer = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    CUT_SAFE_CALL( cutDeleteTimer( timer));

    CUT_EXIT(argc, argv);
    return 0;
}
