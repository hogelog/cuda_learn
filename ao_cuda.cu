#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define WIDTH        256
#define HEIGHT       256
#define NSUBSAMPLES  2
#define NAO_SAMPLES  8

struct vec {
    double x, y, z;
};
struct Isect {
    double t;
    vec p, n;
    int hit; 
};
struct Sphere {
    vec center;
    double radius;
};
struct Plane {
    vec p, n;
};
struct Ray {
    vec org, dir;
};

static double vdot(vec v0, vec v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

static void vcross(vec *c, vec v0, vec v1)
{
    c->x = v0.y * v1.z - v0.z * v1.y;
    c->y = v0.z * v1.x - v0.x * v1.z;
    c->z = v0.x * v1.y - v0.y * v1.x;
}

static void vnormalize(vec *c)
{
    double length = sqrt(vdot((*c), (*c)));

    if (fabs(length) > 1.0e-17) {
        c->x /= length;
        c->y /= length;
        c->z /= length;
    }
}

void
ray_sphere_intersect(Isect *isect, const Ray *ray, const Sphere *sphere)
{
    vec rs = {
        ray->org.x - sphere->center.x,
        ray->org.y - sphere->center.y,
        ray->org.z - sphere->center.z};
    double B = vdot(rs, ray->dir);
    double C = vdot(rs, rs) - sphere->radius * sphere->radius;
    double D = B * B - C;

    if (D > 0.0) {
        double t = -B - sqrt(D);
        
        if ((t > 0.0) && (t < isect->t)) {
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

void
ray_plane_intersect(Isect *isect, const Ray *ray, const Plane *plane)
{
    double d = -vdot(plane->p, plane->n);
    double v = vdot(ray->dir, plane->n);

    if (fabs(v) < 1.0e-17) return;

    double t = -(vdot(ray->org, plane->n) + d) / v;

    if ((t > 0.0) && (t < isect->t)) {
        isect->t = t;
        isect->hit = 1;
        
        isect->p.x = ray->org.x + ray->dir.x * t;
        isect->p.y = ray->org.y + ray->dir.y * t;
        isect->p.z = ray->org.z + ray->dir.z * t;

        isect->n = plane->n;
    }
}

void
orthoBasis(vec *basis, vec n)
{
    basis[2] = n;
    basis[1].x = 0.0; basis[1].y = 0.0; basis[1].z = 0.0;

    if ((n.x < 0.6) && (n.x > -0.6)) {
        basis[1].x = 1.0;
    } else if ((n.y < 0.6) && (n.y > -0.6)) {
        basis[1].y = 1.0;
    } else if ((n.z < 0.6) && (n.z > -0.6)) {
        basis[1].z = 1.0;
    } else {
        basis[1].x = 1.0;
    }

    vcross(&basis[0], basis[1], basis[2]);
    vnormalize(&basis[0]);

    vcross(&basis[1], basis[2], basis[0]);
    vnormalize(&basis[1]);
}


void ambient_occlusion(vec *col, const Isect *isect, Sphere *spheres, Plane plane)
{
    int    ntheta = NAO_SAMPLES;
    int    nphi   = NAO_SAMPLES;
    double eps = 0.0001;

    vec p = {
        isect->p.x + eps * isect->n.x,
        isect->p.y + eps * isect->n.y,
        isect->p.z + eps * isect->n.z};

    vec basis[3];
    orthoBasis(basis, isect->n);

    double occlusion = 0.0;

    for (int j = 0; j < ntheta; j++) {
        for (int i = 0; i < nphi; i++) {
            double theta = sqrt(drand48());
            double phi   = 2.0 * M_PI * drand48();

            double x = cos(phi) * theta;
            double y = sin(phi) * theta;
            double z = sqrt(1.0 - theta * theta);

            // local -> global
            double rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
            double ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
            double rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

            Ray ray = {p, {rx, ry, rz}};
            Isect occIsect = {1.0e+17, {0,0,0}, {0,0,0}, 0};

            ray_sphere_intersect(&occIsect, &ray, &spheres[0]); 
            ray_sphere_intersect(&occIsect, &ray, &spheres[1]); 
            ray_sphere_intersect(&occIsect, &ray, &spheres[2]); 
            ray_plane_intersect (&occIsect, &ray, &plane); 

            if (occIsect.hit) occlusion += 1.0;
        }
    }

    occlusion = (ntheta * nphi - occlusion) / (double)(ntheta * nphi);

    col->x = occlusion;
    col->y = occlusion;
    col->z = occlusion;
}

unsigned char
clamp(double f)
{
  int i = (int)(f * 255.5);

  if (i < 0) i = 0;
  if (i > 255) i = 255;

  return (unsigned char)i;
}


void
render(unsigned char *img, double* fimg, int w, int h, int nsubsamples, Sphere *spheres, Plane plane)
{
    int x, y;
    int u, v;

    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            for (v = 0; v < nsubsamples; v++) {
                for (u = 0; u < nsubsamples; u++) {
                    double px = (x + (u / (double)nsubsamples) - (w / 2.0)) / (w / 2.0);
                    double py = -(y + (v / (double)nsubsamples) - (h / 2.0)) / (h / 2.0);

                    Ray ray = {
                        {0.0, 0.0, 0.0},
                        {px, py, -1.0}};

                    vnormalize(&(ray.dir));

                    Isect isect = {1.0e+17, {0,0,0}, {0,0,0}, 0};

                    ray_sphere_intersect(&isect, &ray, &spheres[0]);
                    ray_sphere_intersect(&isect, &ray, &spheres[1]);
                    ray_sphere_intersect(&isect, &ray, &spheres[2]);
                    ray_plane_intersect (&isect, &ray, &plane);

                    if (isect.hit) {
                        vec col;
                        ambient_occlusion(&col, &isect, spheres, plane);

                        fimg[3 * (y * w + x) + 0] += col.x;
                        fimg[3 * (y * w + x) + 1] += col.y;
                        fimg[3 * (y * w + x) + 2] += col.z;
                    }

                }
            }

            fimg[3 * (y * w + x) + 0] /= (double)(nsubsamples * nsubsamples);
            fimg[3 * (y * w + x) + 1] /= (double)(nsubsamples * nsubsamples);
            fimg[3 * (y * w + x) + 2] /= (double)(nsubsamples * nsubsamples);
            img[3 * (y * w + x) + 0] = clamp(fimg[3 *(y * w + x) + 0]);
            img[3 * (y * w + x) + 1] = clamp(fimg[3 *(y * w + x) + 1]);
            img[3 * (y * w + x) + 2] = clamp(fimg[3 *(y * w + x) + 2]);
        }
    }
}

void
saveppm(const char *fname, int w, int h, unsigned char *img)
{
    FILE *fp = fopen(fname, "wb");
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
    Sphere spheres[3] = {
        {-2.0, 0.0, -3.5, 0.5},
        {-0.5, 0.0, -3.0, 0.5},
        {1.0, 0.0, -2.2, 0.5}};
    Plane plane = {
        {0.0, -0.5, 0.0},
        {0.0, 1.0, 0.0}};
    unsigned char *img = (unsigned char *)malloc(WIDTH * HEIGHT * 3);
    double *fimg = (double *)malloc(sizeof(double) * WIDTH * HEIGHT * 3);

    render(img, fimg, WIDTH, HEIGHT, NSUBSAMPLES, spheres, plane);

    saveppm("ao.ppm", WIDTH, HEIGHT, img); 

    return 0;
}
