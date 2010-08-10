#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH  (256)
#define HEIGHT (256)

#define SCALE (1.0)
#define ENDIAN (-1)

void writeppm(FILE *output, const size_t size) {
    int x, y;
    float *ppmbuf = malloc(sizeof(float)*size);
    for (x=0;x<WIDTH;++x) {
        for (y=0;y<HEIGHT;++y) {
            ppmbuf[y*WIDTH+x] = (float)x / (float)WIDTH;
        }
    }
    fprintf(output, "Pf\n%d %d\n%f\n", WIDTH, HEIGHT, ENDIAN*SCALE);
    fwrite(ppmbuf, sizeof(float), size, output);
    free(ppmbuf);
}

int main() {
    const size_t ppmsize = WIDTH * HEIGHT;
    FILE *ppm = fopen("white2black.ppm", "w");
    if (ppm) {
        writeppm(ppm, ppmsize);
    }
    return 0;
}
