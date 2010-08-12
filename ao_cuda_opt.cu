#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <cutil_inline.h>

#include <thrust/random.h>

#define WIDTH        256
#define HEIGHT       256
#define NSUBSAMPLES  2
#define NAO_SAMPLES  8

#define RASTER_SIZE (sizeof(float)*WIDTH*HEIGHT*1)

#define NSIZE (WIDTH*HEIGHT)

#define NTHREADS 64
#define NBLOCKS (NSIZE/NTHREADS)

typedef thrust::default_random_engine RNG;
typedef thrust::uniform_real_distribution<float> RNG_range;

typedef float3 vec;

struct Isect {
    float t;
    vec p, n;
    float hit; 
};

struct Sphere {
    vec center;
    float radius;
};

struct Plane {
    vec p, n;
};

struct Ray {
    vec org, dir;
};

__constant__ Sphere spheres[3] = {
    {{-2.0f, 0.0f, -3.5f}, 0.5f},
    {{-0.5f, 0.0f, -3.0f}, 0.5f},
    {{1.0f,  0.0f, -2.2f}, 0.5f}};
__constant__ Plane plane = {
    {0.0f, -0.5f, 0.0f},
    {0.0f,  1.0f, 0.0f}};

__device__ inline float
vdot(const vec v0, const vec v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

__device__ inline void
vcross(vec& c, const vec v0, const vec v1)
{
    c.x = v0.y * v1.z - v0.z * v1.y;
    c.y = v0.z * v1.x - v0.x * v1.z;
    c.z = v0.x * v1.y - v0.y * v1.x;
}

__device__ inline void
vnormalize(vec& c)
{
    const float length = sqrt(vdot(c, c));

    if (fabsf(length) > 1.0e-17f) {
        c.x /= length;
        c.y /= length;
        c.z /= length;
    }
}

__device__ void
ray_sphere_intersect(Isect& isect, const Ray& ray, const Sphere& sphere)
{
    const vec rs = {
        ray.org.x - sphere.center.x,
        ray.org.y - sphere.center.y,
        ray.org.z - sphere.center.z};

    const float B = vdot(rs, ray.dir);
    const float C = vdot(rs, rs) - sphere.radius * sphere.radius;
    const float D = B * B - C;

    if (D > 0.0f) {
        const float t = -B - sqrt(D);
        
        if ((t > 0.0f) && (t < isect.t)) {
            isect.t = t;
            isect.hit = 1.0f;
            
            isect.p.x = ray.org.x + ray.dir.x * t;
            isect.p.y = ray.org.y + ray.dir.y * t;
            isect.p.z = ray.org.z + ray.dir.z * t;

            isect.n.x = isect.p.x - sphere.center.x;
            isect.n.y = isect.p.y - sphere.center.y;
            isect.n.z = isect.p.z - sphere.center.z;

            vnormalize(isect.n);
        }
    }
}

__device__ void
ray_plane_intersect(Isect& isect, const Ray& ray)
{
    const float d = -vdot(plane.p, plane.n);
    const float v = vdot(ray.dir, plane.n);

    if (fabsf(v) < 1.0e-17f) return;

    const float t = -(vdot(ray.org, plane.n) + d) / v;

    if ((t > 0.0f) && (t < isect.t)) {
        isect.t = t;
        isect.hit = 1.0f;
        
        isect.p.x = ray.org.x + ray.dir.x * t;
        isect.p.y = ray.org.y + ray.dir.y * t;
        isect.p.z = ray.org.z + ray.dir.z * t;

        isect.n = plane.n;
    }
}

__device__ void
orthoBasis(vec basis[3], vec n)
{
    if ((n.x < 0.6f) && (n.x > -0.6f)) {
        basis[1].x = 1.0f;
    } else if ((n.y < 0.6f) && (n.y > -0.6f)) {
        basis[1].y = 1.0f;
    } else if ((n.z < 0.6f) && (n.z > -0.6f)) {
        basis[1].z = 1.0f;
    } else {
        basis[1].x = 1.0f;
    }

    vcross(basis[0], basis[1], basis[2]);
    vnormalize(basis[0]);

    vcross(basis[1], basis[2], basis[0]);
    vnormalize(basis[1]);
}

__device__ void
ambient_occlusion(vec& col, const Isect& isect, RNG& rng, RNG_range& rng_range)
{
    const float eps = 0.0001f;

    const vec p = {
        isect.p.x + eps * isect.n.x,
        isect.p.y + eps * isect.n.y,
        isect.p.z + eps * isect.n.z};

    vec basis[3] = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        isect.n};
    orthoBasis(basis, isect.n);

    float occlusion = 0.0f;

    for (int j = 0; j < NAO_SAMPLES; j++) {
        for (int i = 0; i < NAO_SAMPLES; i++) {
            const float theta = sqrtf(rng_range(rng));
            const float phi   = 2.0f * (float)M_PI * rng_range(rng);

            const float x = cosf(phi) * theta;
            const float y = sinf(phi) * theta;
            const float z = sqrtf(1.0f - theta * theta);

            // local -> global
            const float rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
            const float ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
            const float rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

            const Ray ray = {
                p,
                {rx, ry, rz}};

            Isect occIsect;
            occIsect.t   = 1.0e+17f;
            occIsect.hit = 0.0f;

            ray_sphere_intersect(occIsect, ray, spheres[0]); 
            ray_sphere_intersect(occIsect, ray, spheres[1]); 
            ray_sphere_intersect(occIsect, ray, spheres[2]); 
            ray_plane_intersect (occIsect, ray); 

            if (occIsect.hit == 1.0f) occlusion += 1.0f;
        }
    }

    occlusion = (NAO_SAMPLES * NAO_SAMPLES - occlusion) / (float)(NAO_SAMPLES * NAO_SAMPLES);

    col.x = occlusion;
    col.y = occlusion;
    col.z = occlusion;
}

__device__ inline float
clamp(float f) {
    if (f < 0.0f) return 0.0f;
    else if (f > 1.0f) return 1.0f;
    else return f;
}

__global__ void
dev_render(float *img)
{
    const int y = blockIdx.x / (NBLOCKS / HEIGHT);
    const int x = threadIdx.x + ((blockIdx.x & (NBLOCKS / HEIGHT - 1)) * NTHREADS);
    const unsigned int seed = y*WIDTH + x;

    // seed a random number generator
    RNG rng(seed);

    // create a mapping from random numbers to [0,1)
    RNG_range rng_range(0, 1);

    float pixel = 0.0f;
    for (int v = 0; v < NSUBSAMPLES; v++) {
        for (int u = 0; u < NSUBSAMPLES; u++) {
            __syncthreads();
            const float px = (x + (u / (float)NSUBSAMPLES) - (WIDTH / 2.0f)) / (WIDTH / 2.0f);
            const float py = -(y + (v / (float)NSUBSAMPLES) - (HEIGHT / 2.0f)) / (HEIGHT / 2.0f);

            Ray ray = {
                {0.0f, 0.0f, 0.0f},
                {px, py, -1.0f}};
            vnormalize(ray.dir);

            Isect isect;
            isect.t   = 1.0e+17f;
            isect.hit = 0.0f;

            ray_sphere_intersect(isect, ray, spheres[0]);
            ray_sphere_intersect(isect, ray, spheres[1]);
            ray_sphere_intersect(isect, ray, spheres[2]);
            ray_plane_intersect (isect, ray);

            if (isect.hit == 1.0f) {
                vec col;
                ambient_occlusion(col, isect, rng, rng_range);

                pixel += (col.x + col.y + col.z) / 3.0f;
            }
        }
    }
    __syncthreads();

    // display(ImageMagic) bug? 
    //img[y * WIDTH + x] = clamp(pixel / (float)(NSUBSAMPLES * NSUBSAMPLES));
    img[(HEIGHT - y) * WIDTH + x] = clamp(pixel / (float)(NSUBSAMPLES * NSUBSAMPLES));
}

void
render(float *img)
{
    float *d_img = NULL;
    cutilSafeCall(cudaMalloc(&d_img, RASTER_SIZE));

    dev_render<<<NBLOCKS, NTHREADS>>>(d_img);
    cutilCheckMsg("dev_render() execution failed");

    cutilSafeCall(cudaMemcpy(img, d_img, RASTER_SIZE, cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaFree(d_img));
}

void
saveppm(const char *fname, float *img)
{
    FILE *fp;

    fp = fopen(fname, "wb");
    assert(fp);

    fprintf(fp, "Pf\n");
    fprintf(fp, "%d %d\n", WIDTH, HEIGHT);
    fprintf(fp, "-1.0\n");
    fwrite(img, sizeof(float), RASTER_SIZE, fp);
    fclose(fp);
}

int
main(int argc, char **argv)
{
    float *img = (float *)malloc(RASTER_SIZE);
    render(img);

    saveppm("ao.ppm", img); 

    return 0;
}
