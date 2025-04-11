
#include "dataloader/load-images-to-disk.c"

int main() {
  char cwd[1024];
  getcwd(cwd, sizeof(cwd));
  printf("Current directory: %s\n", cwd);

  bit_vectors = (uint8_t **)malloc(TOTAL_IMAGES * sizeof(uint8_t *));
  for (int i = 0; i < TOTAL_IMAGES; i++) {
    bit_vectors[i] = (uint8_t *)calloc(BYTES_PER_IMAGE, sizeof(uint8_t));
  }

  const char *dir_path = "./dataset/numerals/";
  DIR *dir = opendir(dir_path);
  if (!dir) {
    perror("Failed to open directory");
    fprintf(stderr, "Tried path: %s\n", dir_path);
    return 1;
  }

  struct dirent *entry;
  int count = 0;
  while ((entry = readdir(dir)) != NULL && count < TOTAL_IMAGES) {
    if (strstr(entry->d_name, ".png")) {
      char path[1024];
      snprintf(path, sizeof(path), "%s%s", dir_path, entry->d_name);

      FILE *test_fp = fopen(path, "rb");
      if (!test_fp) {
        perror("Skipping file (open failed)");
        continue;
      }
      fclose(test_fp);

      printf("Loading image %d: %s\n", count, entry->d_name);
      png_to_bitvector(path, bit_vectors[count]);
      count++;
    }
  }
  closedir(dir);

  printf("Successfully loaded %d images.\n", count);

  uint8_t **downscaled_vectors = malloc(TOTAL_IMAGES * sizeof(uint8_t *));
  for (int i = 0; i < TOTAL_IMAGES; i++) {
    downscaled_vectors[i] = calloc(BYTES_PER_DOWNSCALED_IMAGE, sizeof(uint8_t));
  }

  for (int i = 0; i < count; i++) {
    printf("Downscaling image %d\n", i);
    downscale_bitvector(bit_vectors[i], downscaled_vectors[i]);
  }

  printf("All images downscaled.\n");

  for (int i = 0; i < count; i++)
    free(bit_vectors[i]);
  free(bit_vectors);

  for (int i = 0; i < count; i++)
    free(downscaled_vectors[i]);
  free(downscaled_vectors);

  return 0;
}
