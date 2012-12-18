#include "oil.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Note: almost all of this code is based on (or taken entirely from) code in
   COGL, the awesome C OpenGL wrapper that's managed by the GNOME
   project. That code was, in turn, based on code from Mesa. Spread the GPL
   love! */

/* swaps two float * variables */
#define SWAP_ROWS(a, b) { float *_tmp = (a); (a) = (b); (b) = _tmp; }

static float identity[4][4] = {{1.0f, 0.0f, 0.0f, 0.0f},
                               {0.0f, 1.0f, 0.0f, 0.0f},
                               {0.0f, 0.0f, 1.0f, 0.0f},
                               {0.0f, 0.0f, 0.0f, 1.0f}};

void oil_matrix_set_identity(OILMatrix *matrix) {
    memcpy(matrix->data, identity, sizeof(float) * 16);
}

/* row-major order: if the first row is (1, 2, ...) the memory will
   start out as 1.0f 2.0f ... */
void oil_matrix_set_data(OILMatrix *matrix, const float *data) {
    memcpy(matrix->data, data, sizeof(float) * 16);
}

int oil_matrix_is_identity(const OILMatrix *matrix)
{
    int i;
    for (i = 0; i < 4; i++) {
        if (matrix->data[i][0] - identity[i][0] != 0.0f ||
            matrix->data[i][1] - identity[i][1] != 0.0f ||
            matrix->data[i][2] - identity[i][2] != 0.0f ||
            matrix->data[i][3] - identity[i][3] != 0.0f)
            
            return 0;
    }
    return 1;
}

int oil_matrix_is_zero(const OILMatrix *matrix)
{
    int i;
    for (i = 0; i < 4; i++) {
        if (matrix->data[i][0] != 0.0f ||
            matrix->data[i][1] != 0.0f ||
            matrix->data[i][2] != 0.0f ||
            matrix->data[i][3] != 0.0f)
            
            return 0;
    }
    return 1;
}

/* result == a is allowed, result == b is not */

void oil_matrix_add(OILMatrix *result, const OILMatrix *a, const OILMatrix *b) {
    int i;
    for (i = 0; i < 4; i++) {
        result->data[i][0] = a->data[i][0] + b->data[i][0];
        result->data[i][1] = a->data[i][1] + b->data[i][1];
        result->data[i][2] = a->data[i][2] + b->data[i][2];
        result->data[i][3] = a->data[i][3] + b->data[i][3];
    }
}

void oil_matrix_subtract(OILMatrix *result, const OILMatrix *a, const OILMatrix *b) {
    int i;
    for (i = 0; i < 4; i++) {
        result->data[i][0] = a->data[i][0] - b->data[i][0];
        result->data[i][1] = a->data[i][1] - b->data[i][1];
        result->data[i][2] = a->data[i][2] - b->data[i][2];
        result->data[i][3] = a->data[i][3] - b->data[i][3];
    }
}

void oil_matrix_multiply(OILMatrix *result, const OILMatrix *a, const OILMatrix *b) {
    int i;
    for (i = 0; i < 4; i++) {
        const float a0 = a->data[i][0], a1 = a->data[i][1];
        const float a2 = a->data[i][2], a3 = a->data[i][3];
        result->data[i][0] = a0 * b->data[0][0] + a1 * b->data[1][0] +
            a2 * b->data[2][0] + a3 * b->data[3][0];
        result->data[i][1] = a0 * b->data[0][1] + a1 * b->data[1][1] +
            a2 * b->data[2][1] + a3 * b->data[3][1];
        result->data[i][2] = a0 * b->data[0][2] + a1 * b->data[1][2] +
            a2 * b->data[2][2] + a3 * b->data[3][2];
        result->data[i][3] = a0 * b->data[0][3] + a1 * b->data[1][3] +
            a2 * b->data[2][3] + a3 * b->data[3][3];
    }
}

/* matrix == result allowed */

void oil_matrix_negate(OILMatrix *result, const OILMatrix *matrix)
{
    int i;
    for (i = 0; i < 4; i++) {
        result->data[i][0] = -(matrix->data[i][0]);
        result->data[i][1] = -(matrix->data[i][1]);
        result->data[i][2] = -(matrix->data[i][2]);
        result->data[i][3] = -(matrix->data[i][3]);
    }
}

/* returns 0 on failure */
int oil_matrix_invert(OILMatrix* result, const OILMatrix *matrix) {
    float wtmp[4][8];
    float m0, m1, m2, m3, s;
    float *r0, *r1, *r2, *r3;
    
    r0 = wtmp[0], r1 = wtmp[1], r2 = wtmp[2], r3 = wtmp[3];
    
    memcpy(&(r0[0]), matrix->data[0], sizeof(float) * 4);
    memcpy(&(r0[4]), identity[0], sizeof(float) * 4);

    memcpy(&(r1[0]), matrix->data[1], sizeof(float) * 4);
    memcpy(&(r1[4]), identity[1], sizeof(float) * 4);

    memcpy(&(r2[0]), matrix->data[2], sizeof(float) * 4);
    memcpy(&(r2[4]), identity[2], sizeof(float) * 4);

    memcpy(&(r3[0]), matrix->data[3], sizeof(float) * 4);
    memcpy(&(r3[4]), identity[3], sizeof(float) * 4);
    
    /* choose pivot or die */
    if (fabsf(r3[0]) > fabsf(r2[0]))
        SWAP_ROWS(r3, r2);
    if (fabsf(r2[0]) > fabsf(r1[0]))
        SWAP_ROWS(r2, r1);
    if (fabsf(r1[0]) > fabsf(r0[0]))
        SWAP_ROWS(r1, r0);
    if (r0[0] == 0.0f)
        return 0;
    
    /* eliminate first variable */
    
    m1 = r1[0] / r0[0];
    m2 = r2[0] / r0[0];
    m3 = r3[0] / r0[0];
    
    s = r0[1];
    r1[1] -= m1 * s;
    r2[1] -= m2 * s;
    r3[1] -= m3 * s;
    s = r0[2];
    r1[2] -= m1 * s;
    r2[2] -= m2 * s;
    r3[2] -= m3 * s;
    s = r0[3];
    r1[3] -= m1 * s;
    r2[3] -= m2 * s;
    r3[3] -= m3 * s;
    
    s = r0[4];
    if (s != 0.0f) {
        r1[4] -= m1 * s;
        r2[4] -= m2 * s;
        r3[4] -= m3 * s;
    }
    s = r0[5];
    if (s != 0.0f) {
        r1[5] -= m1 * s;
        r2[5] -= m2 * s;
        r3[5] -= m3 * s;
    }
    s = r0[6];
    if (s != 0.0f) {
        r1[6] -= m1 * s;
        r2[6] -= m2 * s;
        r3[6] -= m3 * s;
    }
    s = r0[7];
    if (s != 0.0f) {
        r1[7] -= m1 * s;
        r2[7] -= m2 * s;
        r3[7] -= m3 * s;
    }
    
    /* choose pivot or die */
    if (fabsf(r3[1]) > fabsf(r2[1]))
        SWAP_ROWS(r3, r2);
    if (fabsf(r2[1]) > fabsf(r1[1]))
        SWAP_ROWS(r2, r1);
    if (r1[1] == 0.0f)
        return 0;
    
    /* eliminate second variable */
    
    m2 = r2[1] / r1[1];
    m3 = r3[1] / r1[1];
    
    s = r1[2];
    r2[2] -= m2 * s;
    r3[2] -= m3 * s;
    s = r1[3];
    r2[3] -= m2 * s;
    r3[3] -= m3 * s;
    
    s = r1[4];
    if (s != 0.0f) {
        r2[4] -= m2 * s;
        r3[4] -= m3 * s;
    }
    s = r1[5];
    if (s != 0.0f) {
        r2[5] -= m2 * s;
        r3[5] -= m3 * s;
    }
    s = r1[6];
    if (s != 0.0f) {
        r2[6] -= m2 * s;
        r3[6] -= m3 * s;
    }
    s = r1[7];
    if (s != 0.0f) {
        r2[7] -= m2 * s;
        r3[7] -= m3 * s;
    }
    
    /* choose pivot or die */
    if (fabsf(r3[2]) > fabsf(r2[2]))
        SWAP_ROWS(r3, r2);
    if (r2[2] == 0.0f)
        return 0;
    
    /* eliminate third variable */
    m3 = r3[2] / r2[2];
    r3[3] -= m3 * r2[3];
    r3[4] -= m3 * r2[4];
    r3[5] -= m3 * r2[5];
    r3[6] -= m3 * r2[6];
    r3[7] -= m3 * r2[7];
    
    /* last check */
    if (r3[3] == 0.0f)
        return 0;
    
    /* now back substitute row 3 */
    s = 1.0f / r3[3];
    r3[4] *= s;
    r3[5] *= s;
    r3[6] *= s;
    r3[7] *= s;
    
    /* back substitute row 2 */
    m2 = r2[3];
    s = 1.0f / r2[2];
    r2[4] = s * (r2[4] - r3[4] * m2);
    r2[5] = s * (r2[5] - r3[5] * m2);
    r2[6] = s * (r2[6] - r3[6] * m2);
    r2[7] = s * (r2[7] - r3[7] * m2);
    m1 = r1[3];
    r1[4] -= r3[4] * m1;
    r1[5] -= r3[5] * m1;
    r1[6] -= r3[6] * m1;
    r1[7] -= r3[7] * m1;
    m0 = r0[3];
    r0[4] -= r3[4] * m0;
    r0[5] -= r3[5] * m0;
    r0[6] -= r3[6] * m0;
    r0[7] -= r3[7] * m0;
    
    /* back substitute row 1 */
    m1 = r1[2];
    s = 1.0f / r1[1];
    r1[4] = s * (r1[4] - r2[4] * m1);
    r1[5] = s * (r1[5] - r2[5] * m1);
    r1[6] = s * (r1[6] - r2[6] * m1);
    r1[7] = s * (r1[7] - r2[7] * m1);
    m0 = r0[2];
    r0[4] -= r2[4] * m0;
    r0[5] -= r2[5] * m0;
    r0[6] -= r2[6] * m0;
    r0[7] -= r2[7] * m0;
    
    /* back substitute row 0 */
    m0 = r0[1];
    s = 1.0f / r0[0];
    r0[4] = s * (r0[4] - r1[4] * m0);
    r0[5] = s * (r0[5] - r1[5] * m0);
    r0[6] = s * (r0[6] - r1[6] * m0);
    r0[7] = s * (r0[7] - r1[7] * m0);
    
    /* copy to output */
    memcpy(result->data[0], &(r0[4]), sizeof(float) * 4);
    memcpy(result->data[1], &(r1[4]), sizeof(float) * 4);
    memcpy(result->data[2], &(r2[4]), sizeof(float) * 4);
    memcpy(result->data[3], &(r3[4]), sizeof(float) * 4);
    
    return 1;
}

void oil_matrix_translate(OILMatrix *matrix, float x, float y, float z)
{
    matrix->data[0][3] = matrix->data[0][0] * x + matrix->data[0][1] * y + matrix->data[0][2] * z + matrix->data[0][3];
    matrix->data[1][3] = matrix->data[1][0] * x + matrix->data[1][1] * y + matrix->data[1][2] * z + matrix->data[1][3];
    matrix->data[2][3] = matrix->data[2][0] * x + matrix->data[2][1] * y + matrix->data[2][2] * z + matrix->data[2][3];
    matrix->data[3][3] = matrix->data[3][0] * x + matrix->data[3][1] * y + matrix->data[3][2] * z + matrix->data[3][3];
}

void oil_matrix_scale(OILMatrix *matrix, float x, float y, float z)
{
    matrix->data[0][0] *= x;
    matrix->data[1][0] *= x;
    matrix->data[2][0] *= x;
    matrix->data[3][0] *= x;

    matrix->data[0][1] *= y;
    matrix->data[1][1] *= y;
    matrix->data[2][1] *= y;
    matrix->data[3][1] *= y;

    matrix->data[0][2] *= z;
    matrix->data[1][2] *= z;
    matrix->data[2][2] *= z;
    matrix->data[3][2] *= z;
}

void oil_matrix_rotate(OILMatrix *matrix, float x, float y, float z) {
    OILMatrix tmp;
    int optimized = 0;
    float c, s;
    
    oil_matrix_set_identity(&tmp);
        
    if (x == 0.0f) {
        if (y == 0.0f) {
            if (z == 0.0f) {
                /* no rotation, we're already done */
                return;
            } else {
                optimized = 1;
                
                /* rotate only around z-axis */
                c = cosf(z);
                s = sinf(z);
                
                tmp.data[0][0] = c;
                tmp.data[1][1] = c;
                tmp.data[0][1] = -s;
                tmp.data[1][0] = s;
            }
        } else if (z == 0.0f) {
            optimized = 1;
            
            /* rotate only around y axis */
            c = cosf(y);
            s = sinf(y);
            
            tmp.data[0][0] = c;
            tmp.data[2][2] = c;
            tmp.data[0][2] = s;
            tmp.data[2][0] = -s;
        }
    } else if (y == 0.0f && z == 0.0f) {
        optimized = 1;
        
        /* rotate only around x axis */
        c = cosf(x);
        s = sinf(x);
        
        tmp.data[1][1] = c;
        tmp.data[2][2] = c;
        tmp.data[1][2] = -s;
        tmp.data[2][1] = s;
    }
    
    if (!optimized) {
        float xx, yy, zz;
        float xy, yz, zx;
        float xs, ys, zs;
        float one_c;
        
        const float mag = sqrtf(x * x + y * y + z * z);
        x /= mag;
        y /= mag;
        z /= mag;
        
        c = cosf(mag);
        s = sinf(mag);
        
        xx = x * x;
        yy = y * y;
        zz = z * z;
        xy = x * y;
        yz = y * z;
        zx = z * x;
        xs = x * s;
        ys = y * s;
        zs = z * s;
        one_c = 1.0f - c;
        
        tmp.data[0][0] = (one_c * xx) + c;
        tmp.data[0][1] = (one_c * xy) - zs;
        tmp.data[0][2] = (one_c * zx) + ys;
        
        tmp.data[1][0] = (one_c * xy) + zs;
        tmp.data[1][1] = (one_c * yy) + c;
        tmp.data[1][2] = (one_c * yz) - xs;
        
        tmp.data[2][0] = (one_c * zx) - ys;
        tmp.data[2][1] = (one_c * yz) + xs;
        tmp.data[2][2] = (one_c * zz) + c;
    }
    
    oil_matrix_multiply(matrix, matrix, &tmp);
}

void oil_matrix_orthographic(OILMatrix *matrix, float x1, float x2, float y1, float y2, float z1, float z2) {
    OILMatrix tmp;
    oil_matrix_set_identity(&tmp);
    
    tmp.data[0][0] = 2.0f / (x2 - x1);
    tmp.data[0][3] = -(x2 + x1) / (x2 - x1);
    
    tmp.data[1][1] = 2.0f / (y2 - y1);
    tmp.data[1][3] = -(y2 + y1) / (y2 - y1);
    
    tmp.data[2][2] = 2.0f / (z2 - z1);
    tmp.data[2][3] = -(z2 + z1) / (z2 - z1);
    
    oil_matrix_multiply(matrix, matrix, &tmp);
}
