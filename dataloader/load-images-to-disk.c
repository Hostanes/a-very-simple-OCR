#include <dirent.h>
#include <png.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define WIDTH 1200
#define HEIGHT 900
#define TOTAL_IMAGES 550
#define PIXELS_PER_IMAGE (WIDTH * HEIGHT)
#define BYTES_PER_IMAGE ((PIXELS_PER_IMAGE + 7) / 8)

#define DOWNSCALED_WIDTH 400
#define DOWNSCALED_HEIGHT 300
#define PIXELS_PER_DOWNSCALED_IMAGE (DOWNSCALED_WIDTH * DOWNSCALED_HEIGHT)
#define BYTES_PER_DOWNSCALED_IMAGE ((PIXELS_PER_DOWNSCALED_IMAGE + 7) / 8)

uint8_t **bit_vectors;

void bit_to_double(int img_idx, double *output) {
  for (size_t i = 0; i < PIXELS_PER_IMAGE; i++) {
    uint8_t byte = bit_vectors[img_idx][i / 8];
    uint8_t bit = (byte >> (i % 8)) & 1;
    output[i] = (double)bit;
  }
}

void png_to_bitvector(const char *filename, uint8_t *vector) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Failed to open: %s\n", filename);
    exit(1);
  }

  png_structp png =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = png_create_info_struct(png);
  if (!png || !info)
    exit(1);

  if (setjmp(png_jmpbuf(png)))
    exit(1);
  png_init_io(png, fp);
  png_read_info(png, info);

  if (png_get_image_width(png, info) != WIDTH ||
      png_get_image_height(png, info) != HEIGHT) {
    fprintf(stderr, "Error: Image %s is not 1200x900\n", filename);
    exit(1);
  }

  png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * HEIGHT);
  for (int y = 0; y < HEIGHT; y++) {
    row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png, info));
  }
  png_read_image(png, row_pointers);

  // DEBUG: Check average brightness
  int white_pixels = 0;

  for (int y = 0; y < HEIGHT; y++) {
    png_bytep row = row_pointers[y];
    for (int x = 0; x < WIDTH; x++) {
      png_bytep px = &(row[x * 4]); // RGBA
      bool is_white = (px[0] + px[1] + px[2] > 384);
      size_t pixel_idx = y * WIDTH + x;
      size_t byte_idx = pixel_idx / 8;
      uint8_t bit_pos = pixel_idx % 8;
      if (is_white) {
        vector[byte_idx] |= (1 << bit_pos);
        white_pixels++;
      } else {
        vector[byte_idx] &= ~(1 << bit_pos);
      }
    }
    free(row_pointers[y]);
  }
  free(row_pointers);
  png_destroy_read_struct(&png, &info, NULL);
  fclose(fp);

  // DEBUG:
  printf("Converted %s to bit vector. Approx white pixel count: %d\n", filename,
         white_pixels);
}

void downscale_bitvector(uint8_t *input, uint8_t *output) {
  memset(output, 0, BYTES_PER_DOWNSCALED_IMAGE);

  for (int y = 0; y < DOWNSCALED_HEIGHT; y++) {
    for (int x = 0; x < DOWNSCALED_WIDTH; x++) {
      int white_count = 0;

      for (int dy = 0; dy < 3; dy++) {
        for (int dx = 0; dx < 3; dx++) {
          int src_x = x * 3 + dx;
          int src_y = y * 3 + dy;
          int src_idx = src_y * WIDTH + src_x;
          int byte = input[src_idx / 8];
          int bit = (byte >> (src_idx % 8)) & 1;
          white_count += bit;
        }
      }

      if (white_count >= 5) {
        int dst_idx = y * DOWNSCALED_WIDTH + x;
        output[dst_idx / 8] |= (1 << (dst_idx % 8));
      }
    }
  }

  // DEBUG:
  int total_white = 0;
  for (int i = 0; i < PIXELS_PER_DOWNSCALED_IMAGE; i++) {
    if ((output[i / 8] >> (i % 8)) & 1)
      total_white++;
  }
  printf("Downscaled image has approx %d white pixels.\n", total_white);
}


