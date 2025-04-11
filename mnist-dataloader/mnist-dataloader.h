/*
  A modified implementation of
  MNIST loader by Nuri Park - https://github.com/projectgalateia/mnist
  modifications:
  1. Pixels are read as binary values of 0 and 1
  2. Output format is flattened to 1D vectors (784 elements)
*/

#ifndef __MNIST_H__
#define __MNIST_H__

#ifdef USE_MNIST_LOADER /* Fundamental macro to make the code active */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Make mnist_load function static.
 * Define when the header is included multiple time.
 */
#ifdef MNIST_STATIC
#define _STATIC static
#else
#define _STATIC
#endif

/*
 * Data type options
 */
#ifdef MNIST_BINARY
#define MNIST_DATA_TYPE unsigned char /* Binary mode uses 0/1 values */
#elif defined(MNIST_DOUBLE)
#define MNIST_DATA_TYPE double
#else
#define MNIST_DATA_TYPE unsigned char
#endif

/*
 * Modified data structure with 1D array
 */
typedef struct mnist_data {
  MNIST_DATA_TYPE data[784]; /* Flattened 28x28 image (784 elements) */
  unsigned int label;        /* label : 0 to 9 */
} mnist_data;

#ifdef MNIST_HDR_ONLY
_STATIC int mnist_load(const char *image_filename, const char *label_filename,
                       mnist_data **data, unsigned int *count);
#else

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Helper function to read 4-byte integers from file (MSB first) */
static unsigned int mnist_bin_to_int(char *v) {
  unsigned int ret = 0;
  for (int i = 0; i < 4; ++i) {
    ret <<= 8;
    ret |= (unsigned char)v[i];
  }
  return ret;
}

/*
 * MNIST dataset loader (modified for 1D output)
 *
 * Returns 0 on success, negative values on error:
 * -1: File not found
 * -2: Invalid image file
 * -3: Invalid label file
 * -4: Image/label count mismatch
 */
_STATIC int mnist_load(const char *image_filename, const char *label_filename,
                       mnist_data **data, unsigned int *count) {
  int return_code = 0;
  FILE *ifp = NULL, *lfp = NULL;
  char tmp[4];

  /* Open files */
  ifp = fopen(image_filename, "rb");
  lfp = fopen(label_filename, "rb");
  if (!ifp || !lfp) {
    return_code = -1;
    goto cleanup;
  }

  /* Verify file magic numbers */
  fread(tmp, 1, 4, ifp);
  if (mnist_bin_to_int(tmp) != 2051) {
    return_code = -2;
    goto cleanup;
  }

  fread(tmp, 1, 4, lfp);
  if (mnist_bin_to_int(tmp) != 2049) {
    return_code = -3;
    goto cleanup;
  }

  /* Read counts */
  fread(tmp, 1, 4, ifp);
  unsigned int image_cnt = mnist_bin_to_int(tmp);
  fread(tmp, 1, 4, lfp);
  unsigned int label_cnt = mnist_bin_to_int(tmp);

  if (image_cnt != label_cnt) {
    return_code = -4;
    goto cleanup;
  }

  /* Verify image dimensions */
  fread(tmp, 1, 4, ifp); // Skip dimensions (we know they're 28x28)

  *count = image_cnt;
  *data = (mnist_data *)malloc(sizeof(mnist_data) * image_cnt);

  /* Read all images */
  for (unsigned int i = 0; i < image_cnt; ++i) {
    unsigned char read_data[28 * 28];
    mnist_data *d = &(*data)[i];

    /* Read image data */
    fread(read_data, 1, 28 * 28, ifp);

    /* Process pixels */
#ifdef MNIST_BINARY
    for (int j = 0; j < 28 * 28; ++j) {
      d->data[j] = (read_data[j] > 1) ? 1 : 0;
    }
#elif defined(MNIST_DOUBLE)
    for (int j = 0; j < 28 * 28; ++j) {
      d->data[j] = read_data[j] / 255.0;
    }
#else
    memcpy(d->data, read_data, 28 * 28);
#endif

    /* Read label */
    fread(tmp, 1, 1, lfp);
    d->label = tmp[0];
  }

cleanup:
  if (ifp)
    fclose(ifp);
  if (lfp)
    fclose(lfp);
  return return_code;
}

#endif /* MNIST_HDR_ONLY */

#ifdef __cplusplus
}
#endif

#endif /* USE_MNIST_LOADER */
#endif /* __MNIST_H__ */

 
   
