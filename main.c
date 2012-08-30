#include "oil.h"
#include <stdio.h>

int main(int argc, char **argv) {
    OILImage *im = oil_image_load("input.png");
    if (!im) {
        printf("failed to load\n");
        return 1;
    }
    if (!oil_image_save(im, "test.png")) {
        printf("failed to save\n");
        return 1;
    }
    oil_image_free(im);
    printf("success!\n");
    return 0;
}
