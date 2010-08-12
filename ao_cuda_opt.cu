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

#define RASTER_SIZE (sizeof(float)*WIDTH*HEIGHT*3)

#define NTHREADS 64
#define NBLOCKS (WIDTH*HEIGHT/NTHREADS)

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

__device__ static float
vdot(vec v0, vec v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

__device__ static void
vcross(vec& c, vec v0, vec v1)
{
    c.x = v0.y * v1.z - v0.z * v1.y;
    c.y = v0.z * v1.x - v0.x * v1.z;
    c.z = v0.x * v1.y - v0.y * v1.x;
}

__device__ static void
vnormalize(vec& c)
{
    float length = sqrt(vdot(c, c));

    if (fabs(length) > 1.0e-17f) {
        c.x /= length;
        c.y /= length;
        c.z /= length;
    }
}

__device__ void
ray_sphere_intersect(Isect& isect, const Ray& ray, const Sphere& sphere)
{
    vec rs;

    rs.x = ray.org.x - sphere.center.x;
    rs.y = ray.org.y - sphere.center.y;
    rs.z = ray.org.z - sphere.center.z;

    float B = vdot(rs, ray.dir);
    float C = vdot(rs, rs) - sphere.radius * sphere.radius;
    float D = B * B - C;

    if (D > 0.0f) {
        float t = -B - sqrt(D);
        
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
    float d = -vdot(plane.p, plane.n);
    float v = vdot(ray.dir, plane.n);

    if (fabs(v) < 1.0e-17f) return;

    float t = -(vdot(ray.org, plane.n) + d) / v;

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
orthoBasis(vec *basis, vec n)
{
    basis[2] = n;
    basis[1].x = 0.0f; basis[1].y = 0.0f; basis[1].z = 0.0f;

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
ambient_occlusion(vec *col, const Isect *isect, RNG& rng, RNG_range& rng_range)
{
    const float eps = 0.0001f;

    vec p;

    p.x = isect->p.x + eps * isect->n.x;
    p.y = isect->p.y + eps * isect->n.y;
    p.z = isect->p.z + eps * isect->n.z;

    vec basis[3];
    orthoBasis(basis, isect->n);

    float occlusion = 0.0f;

    for (int j = 0; j < NAO_SAMPLES; j++) {
        for (int i = 0; i < NAO_SAMPLES; i++) {
            float theta = sqrt(rng_range(rng));
            float phi   = 2.0f * (float)M_PI * rng_range(rng);

            float x = cos(phi) * theta;
            float y = sin(phi) * theta;
            float z = sqrt(1.0f - theta * theta);

            // local -> global
            float rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
            float ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
            float rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

            Ray ray;

            ray.org = p;
            ray.dir.x = rx;
            ray.dir.y = ry;
            ray.dir.z = rz;

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

    col->x = occlusion;
    col->y = occlusion;
    col->z = occlusion;
}

__device__ float
clamp(float f) {
    if (f < 0.0f) return 0.0f;
    else if (f > 1.0f) return 1.0f;
    else return f;
}

__global__ void
dev_render(float *img)
{
    int y = blockIdx.x / (NBLOCKS / HEIGHT);
    int x = threadIdx.x + ((blockIdx.x & (NBLOCKS / HEIGHT - 1)) * NTHREADS);
    unsigned int seed = y*WIDTH + x;

    // seed a random number generator
    RNG rng(seed);

    // create a mapping from random numbers to [0,1)
    RNG_range rng_range(0, 1);

    vec pixel = {0.0f, 0.0f, 0.0f};
    for (int v = 0; v < NSUBSAMPLES; v++) {
        for (int u = 0; u < NSUBSAMPLES; u++) {
            float px = (x + (u / (float)NSUBSAMPLES) - (WIDTH / 2.0f)) / (WIDTH / 2.0f);
            float py = -(y + (v / (float)NSUBSAMPLES) - (HEIGHT / 2.0f)) / (HEIGHT / 2.0f);

            Ray ray;

            ray.org.x = 0.0f;
            ray.org.y = 0.0f;
            ray.org.z = 0.0f;

            ray.dir.x = px;
            ray.dir.y = py;
            ray.dir.z = -1.0f;
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
                ambient_occlusion(&col, &isect, rng, rng_range);

                pixel.x += col.x;
                pixel.y += col.y;
                pixel.z += col.z;
            }
        }
    }
    pixel.x /= (float)(NSUBSAMPLES * NSUBSAMPLES);
    pixel.y /= (float)(NSUBSAMPLES * NSUBSAMPLES);
    pixel.z /= (float)(NSUBSAMPLES * NSUBSAMPLES);
    // display(ImageMagic) bug? 
    //img[3 * (y * WIDTH + x) + 0] = clamp(pixel.x);
    //img[3 * (y * WIDTH + x) + 1] = clamp(pixel.y);
    //img[3 * (y * WIDTH + x) + 2] = clamp(pixel.z);
    img[3 * ((HEIGHT - y) * WIDTH + x) + 0] = clamp(pixel.x);
    img[3 * ((HEIGHT - y) * WIDTH + x) + 1] = clamp(pixel.y);
    img[3 * ((HEIGHT - y) * WIDTH + x) + 2] = clamp(pixel.z);
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

    fprintf(fp, "PF\n");
    fprintf(fp, "%d %d\n", WIDTH, HEIGHT);
    fprintf(fp, "-1.0\n");
    fwrite(img, RASTER_SIZE, 1, fp);
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
