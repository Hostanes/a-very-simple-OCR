#include "lib/nnlib.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WINDOW_WIDTH 560 // 28 * 20 (scaling factor)
#define WINDOW_HEIGHT 560
#define GRID_SIZE 28
#define CELL_SIZE (WINDOW_WIDTH / GRID_SIZE)
#define BUTTON_HEIGHT 40
#define INPUT_SIZE (GRID_SIZE * GRID_SIZE)

// Global pixel data (0=white, 1=black)
float pixels[GRID_SIZE][GRID_SIZE] = {0};
NeuralNetwork_t *model = NULL;

typedef struct {
  SDL_Rect rect;
  const char *label;
  bool hovered;
} Button;

Button guess_button = {{0, WINDOW_HEIGHT, 200, BUTTON_HEIGHT}, "Guess", false};
Button reset_button = {
    {200, WINDOW_HEIGHT, 200, BUTTON_HEIGHT}, "Reset", false};
Button load_button = {
    {400, WINDOW_HEIGHT, 160, BUTTON_HEIGHT}, "Load Model", false};

void initialize_pixels() {
  for (int i = 0; i < GRID_SIZE; i++) {
    for (int j = 0; j < GRID_SIZE; j++) {
      pixels[i][j] = 0.0f;
    }
  }
}

void render_grid(SDL_Renderer *renderer, TTF_Font *font) {
  // Clear screen
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
  SDL_RenderClear(renderer);

  // Draw grid cells
  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      // Calculate grayscale value (0-255)
      Uint8 gray = (Uint8)((1.0 - pixels[y][x]) * 255);
      SDL_SetRenderDrawColor(renderer, gray, gray, gray, 255);

      SDL_Rect cell = {x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE};
      SDL_RenderFillRect(renderer, &cell);
    }
  }

  // Draw grid lines
  SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
  for (int i = 0; i <= GRID_SIZE; i++) {
    // Vertical lines
    SDL_RenderDrawLine(renderer, i * CELL_SIZE, 0, i * CELL_SIZE,
                       WINDOW_HEIGHT);
    // Horizontal lines
    SDL_RenderDrawLine(renderer, 0, i * CELL_SIZE, WINDOW_WIDTH, i * CELL_SIZE);
  }

  // Draw buttons
  Button buttons[] = {guess_button, reset_button, load_button};
  for (int i = 0; i < 3; i++) {
    Button btn = buttons[i];

    // Button background
    SDL_SetRenderDrawColor(renderer, btn.hovered ? 200 : 230, 230, 230, 255);
    SDL_RenderFillRect(renderer, &btn.rect);

    // Button border
    SDL_SetRenderDrawColor(renderer, 150, 150, 150, 255);
    SDL_RenderDrawRect(renderer, &btn.rect);

    // Button text
    if (font) {
      SDL_Color textColor = {0, 0, 0, 255};
      SDL_Surface *textSurface =
          TTF_RenderText_Blended(font, btn.label, textColor);
      SDL_Texture *textTexture =
          SDL_CreateTextureFromSurface(renderer, textSurface);

      SDL_Rect textRect = {btn.rect.x + (btn.rect.w - textSurface->w) / 2,
                           btn.rect.y + (btn.rect.h - textSurface->h) / 2,
                           textSurface->w, textSurface->h};

      SDL_RenderCopy(renderer, textTexture, NULL, &textRect);

      SDL_FreeSurface(textSurface);
      SDL_DestroyTexture(textTexture);
    }
  }

  SDL_RenderPresent(renderer);
}

void classify_image() {
  if (!model) {
    printf("No model loaded!\n");
    return;
  }

  // Prepare input for network (flatten the 28x28 array)
  float input[INPUT_SIZE];
  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      input[y * GRID_SIZE + x] = pixels[y][x];
    }
  }

  // Get prediction
  float *output = forward_pass(model, input);
  int prediction = predict(model, input);
  float confidence = output[prediction] * 100;

  printf("\nClassification Result:\n");
  printf("Predicted digit: %d (confidence: %.2f%%)\n", prediction, confidence);
  printf("Output values:\n");
  for (int i = 0; i < 10; i++) {
    printf("%d: %.4f\n", i, output[i]);
  }
}

void load_model() {
  if (model) {
    free_network(model);
  }
  model = load_Network("serial.nn");
  if (model) {
    printf("Model loaded successfully!\n");
  } else {
    printf("Failed to load model!\n");
  }
}

int main(int argc, char *argv[]) {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
    return 1;
  }

  // Initialize TTF for button text
  if (TTF_Init() < 0) {
    printf("SDL_ttf could not initialize! TTF_Error: %s\n", TTF_GetError());
    SDL_Quit();
    return 1;
  }

  TTF_Font *font = TTF_OpenFont("arial.ttf", 24);
  if (!font) {
    printf("Failed to load font! Using button labels without text.\n");
  }

  SDL_Window *window =
      SDL_CreateWindow("MNIST Drawing Classifier", SDL_WINDOWPOS_UNDEFINED,
                       SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH,
                       WINDOW_HEIGHT + BUTTON_HEIGHT, SDL_WINDOW_SHOWN);

  if (!window) {
    printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
    TTF_Quit();
    SDL_Quit();
    return 1;
  }

  SDL_Renderer *renderer =
      SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  if (!renderer) {
    printf("Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();
    return 1;
  }

  initialize_pixels();
  bool running = true;
  bool mouse_down = false;

  while (running) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      switch (event.type) {
      case SDL_QUIT:
        running = false;
        break;

      case SDL_MOUSEBUTTONDOWN:
        if (event.button.button == SDL_BUTTON_LEFT) {
          mouse_down = true;

          // Check if a button was clicked
          SDL_Point mouse_pos = {event.button.x, event.button.y};

          if (SDL_PointInRect(&mouse_pos, &guess_button.rect)) {
            classify_image();
          } else if (SDL_PointInRect(&mouse_pos, &reset_button.rect)) {
            initialize_pixels();
          } else if (SDL_PointInRect(&mouse_pos, &load_button.rect)) {
            load_model();
          } else if (mouse_pos.y < WINDOW_HEIGHT) {
            // Handle drawing on the grid
            int mouse_x = event.button.x / CELL_SIZE;
            int mouse_y = event.button.y / CELL_SIZE;

            if (mouse_x >= 0 && mouse_x < GRID_SIZE && mouse_y >= 0 &&
                mouse_y < GRID_SIZE) {
              pixels[mouse_y][mouse_x] =
                  fmin(1.0, pixels[mouse_y][mouse_x] + 0.2);
            }
          }
        }
        break;

      case SDL_MOUSEMOTION:
        if (mouse_down) {
          SDL_Point mouse_pos = {event.motion.x, event.motion.y};

          // Only draw if we're in the grid area
          if (mouse_pos.y < WINDOW_HEIGHT) {
            int mouse_x = mouse_pos.x / CELL_SIZE;
            int mouse_y = mouse_pos.y / CELL_SIZE;

            if (mouse_x >= 0 && mouse_x < GRID_SIZE && mouse_y >= 0 &&
                mouse_y < GRID_SIZE) {
              for (int i_loc = -1; i_loc <= 1; i_loc++) {
                for (int j_loc = -1; j_loc <= 1; j_loc++) {
                  pixels[mouse_y + i_loc][mouse_x + j_loc] =
                      fmin(1.0, pixels[mouse_y + i_loc][mouse_x + j_loc] + 0.5);
                }
              }
            }
          }
        }

        // Update button hover states
        SDL_Point mouse_pos = {event.motion.x, event.motion.y};
        guess_button.hovered = SDL_PointInRect(&mouse_pos, &guess_button.rect);
        reset_button.hovered = SDL_PointInRect(&mouse_pos, &reset_button.rect);
        load_button.hovered = SDL_PointInRect(&mouse_pos, &load_button.rect);
        break;

      case SDL_MOUSEBUTTONUP:
        if (event.button.button == SDL_BUTTON_LEFT) {
          mouse_down = false;
        }
        break;
      }
    }

    render_grid(renderer, font);
    SDL_Delay(16); // ~60 FPS
  }

  if (model) {
    free_network(model);
  }
  if (font) {
    TTF_CloseFont(font);
  }
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  TTF_Quit();
  SDL_Quit();

  return 0;
}
