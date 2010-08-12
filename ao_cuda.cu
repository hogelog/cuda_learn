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

typedef thrust::default_random_engine RNG;
typedef thrust::uniform_real_distribution<float> RNG_range;

struct vec
{
    float x;
    float y;
    float z;
};


struct Isect
{
    float t;
    vec    p;
    vec    n;
    int    hit; 
};

struct Sphere
{
    vec    center;
    float radius;

};

struct Plane
{
    vec    p;
    vec    n;
};

struct Ray
{
    vec    org;
    vec    dir;
};

__device__ Sphere spheres[3] = {
    {{-2.0f, 0.0f, -3.5f}, 0.5f},
    {{-0.5f, 0.0f, -3.0f}, 0.5f},
    {{1.0f,  0.0f, -2.2f}, 0.5f}};
__device__ Plane plane = {
    {0.0f, -0.5f, 0.0f},
    {0.0f,  1.0f, 0.0f}};

__device__ static float
vdot(vec v0, vec v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

__device__ static void
vcross(vec *c, vec v0, vec v1)
{
    
    c->x = v0.y * v1.z - v0.z * v1.y;
    c->y = v0.z * v1.x - v0.x * v1.z;
    c->z = v0.x * v1.y - v0.y * v1.x;
}

__device__ static void
vnormalize(vec *c)
{
    float length = sqrt(vdot((*c), (*c)));

    if (fabs(length) > 1.0e-17f) {
        c->x /= length;
        c->y /= length;
        c->z /= length;
    }
}

__device__ void
ray_sphere_intersect(Isect *isect, const Ray *ray, const Sphere *sphere)
{
    vec rs;

    rs.x = ray->org.x - sphere->center.x;
    rs.y = ray->org.y - sphere->center.y;
    rs.z = ray->org.z - sphere->center.z;

    float B = vdot(rs, ray->dir);
    float C = vdot(rs, rs) - sphere->radius * sphere->radius;
    float D = B * B - C;

    if (D > 0.0f) {
        float t = -B - sqrt(D);
        
        if ((t > 0.0f) && (t < isect->t)) {
            isect->t = t;
            isect->hit = 1;
            
            isect->p.x = ray->org.x + ray->dir.x * t;
            isect->p.y = ray->org.y + ray->dir.y * t;
            isect->p.z = ray->org.z + ray->dir.z * t;

            isect->n.x = isect->p.x - sphere->center.x;
            isect->n.y = isect->p.y - sphere->center.y;
            isect->n.z = isect->p.z - sphere->center.z;

            vnormalize(&(isect->n));
        }
    }
}

__device__ void
ray_plane_intersect(Isect *isect, const Ray *ray, const Plane *plane)
{
    float d = -vdot(plane->p, plane->n);
    float v = vdot(ray->dir, plane->n);

    if (fabs(v) < 1.0e-17f) return;

    float t = -(vdot(ray->org, plane->n) + d) / v;

    if ((t > 0.0f) && (t < isect->t)) {
        isect->t = t;
        isect->hit = 1;
        
        isect->p.x = ray->org.x + ray->dir.x * t;
        isect->p.y = ray->org.y + ray->dir.y * t;
        isect->p.z = ray->org.z + ray->dir.z * t;

        isect->n = plane->n;
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

    vcross(&basis[0], basis[1], basis[2]);
    vnormalize(&basis[0]);

    vcross(&basis[1], basis[2], basis[0]);
    vnormalize(&basis[1]);
}

__device__ void
ambient_occlusion(vec *col, const Isect *isect, RNG& rng, RNG_range& rng_range)
{
    int    i, j;
    int    ntheta = NAO_SAMPLES;
    int    nphi   = NAO_SAMPLES;
    float eps = 0.0001f;

    vec p;

    p.x = isect->p.x + eps * isect->n.x;
    p.y = isect->p.y + eps * isect->n.y;
    p.z = isect->p.z + eps * isect->n.z;

    vec basis[3];
    orthoBasis(basis, isect->n);

    float occlusion = 0.0f;

    for (j = 0; j < ntheta; j++) {
        for (i = 0; i < nphi; i++) {
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
            occIsect.hit = 0;

            ray_sphere_intersect(&occIsect, &ray, &spheres[0]); 
            ray_sphere_intersect(&occIsect, &ray, &spheres[1]); 
            ray_sphere_intersect(&occIsect, &ray, &spheres[2]); 
            ray_plane_intersect (&occIsect, &ray, &plane); 

            if (occIsect.hit) occlusion += 1.0f;
            
        }
    }

    occlusion = (ntheta * nphi - occlusion) / (float)(ntheta * nphi);

    col->x = occlusion;
    col->y = occlusion;
    col->z = occlusion;
}

__global__ void
dev_render(float *fimg, int w, int h, int nsubsamples)
{
    int x = threadIdx.x;
    int y = blockIdx.x;
    unsigned int seed = y*w + x;

    // seed a random number generator
    RNG rng(seed);

    // create a mapping from random numbers to [0,1)
    RNG_range rng_range(0,1);

    for (int v = 0; v < nsubsamples; v++) {
        for (int u = 0; u < nsubsamples; u++) {
            float px = (x + (u / (float)nsubsamples) - (w / 2.0f)) / (w / 2.0f);
            float py = -(y + (v / (float)nsubsamples) - (h / 2.0f)) / (h / 2.0f);

            Ray ray;

            ray.org.x = 0.0f;
            ray.org.y = 0.0f;
            ray.org.z = 0.0f;

            ray.dir.x = px;
            ray.dir.y = py;
            ray.dir.z = -1.0f;
            vnormalize(&(ray.dir));

            Isect isect;
            isect.t   = 1.0e+17f;
            isect.hit = 0;

            ray_sphere_intersect(&isect, &ray, &spheres[0]);
            ray_sphere_intersect(&isect, &ray, &spheres[1]);
            ray_sphere_intersect(&isect, &ray, &spheres[2]);
            ray_plane_intersect (&isect, &ray, &plane);

            if (isect.hit) {
                vec col;
                ambient_occlusion(&col, &isect, rng, rng_range);

                fimg[3 * (y * w + x) + 0] += col.x;
                fimg[3 * (y * w + x) + 1] += col.y;
                fimg[3 * (y * w + x) + 2] += col.z;
            }

        }
    }

    fimg[3 * (y * w + x) + 0] /= (float)(nsubsamples * nsubsamples);
    fimg[3 * (y * w + x) + 1] /= (float)(nsubsamples * nsubsamples);
    fimg[3 * (y * w + x) + 2] /= (float)(nsubsamples * nsubsamples);
}

unsigned char
clamp(float f)
{
  int i = (int)(f * 255.5f);

  if (i < 0) i = 0;
  if (i > 255) i = 255;

  return (unsigned char)i;
}

void
render(unsigned char *img, int w, int h, int nsubsamples)
{
    float *fimg = (float *)malloc(sizeof(float) * w * h * 3);
    float *d_fimg = NULL;
    cutilSafeCall(cudaMalloc(&d_fimg, sizeof(float)*w*h*3));

    dev_render<<<w,h>>>(d_fimg, w, h, nsubsamples);
    cutilCheckMsg("dev_render() execution failed\n");

    cutilSafeCall(cudaMemcpy(fimg, d_fimg, sizeof(float)*w*h*3, cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaFree(d_fimg));
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            img[3 * (y * w + x) + 0] = clamp(fimg[3 *(y * w + x) + 0]);
            img[3 * (y * w + x) + 1] = clamp(fimg[3 *(y * w + x) + 1]);
            img[3 * (y * w + x) + 2] = clamp(fimg[3 *(y * w + x) + 2]);
        }
    }
    free(fimg);
}

void
saveppm(const char *fname, int w, int h, unsigned char *img)
{
    FILE *fp;

    fp = fopen(fname, "wb");
    assert(fp);

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", w, h);
    fprintf(fp, "255\n");
    fwrite(img, w * h * 3, 1, fp);
    fclose(fp);
}

int
main(int argc, char **argv)
{
    unsigned char *img = (unsigned char *)malloc(WIDTH * HEIGHT * 3);

    render(img, WIDTH, HEIGHT, NSUBSAMPLES);

    saveppm("ao.ppm", WIDTH, HEIGHT, img); 

    return 0;
}
