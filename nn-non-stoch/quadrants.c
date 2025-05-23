
#include "lib/nnlib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Image dimensions and network architecture
#define IMG_SIZE 8  // 8x8 images
#define INPUT_SIZE (IMG_SIZE * IMG_SIZE)  // 64 input neurons
#define H1_SIZE 16  // First hidden layer
#define H2_SIZE 16  // Second hidden layer
#define OUTPUT_SIZE 4  // 4 classes (quadrants)
#define NUM_SAMPLES 1000  // Number of training samples per epoch
#define NUM_EPOCHS 20

// Layer sizes array
int layer_sizes[] = {INPUT_SIZE, H1_SIZE, H2_SIZE, OUTPUT_SIZE};

// Network parameters and buffers
float *weights = NULL;
float *biases = NULL;
float *Z_values = NULL;
float *A_values = NULL;
float *dZ_values = NULL;
float *dW = NULL;
float *db = NULL;
float *dA = NULL;
int *weight_offsets = NULL;
int *activation_offsets = NULL;

// Function to generate a synthetic image with a bright pixel in a specific quadrant
void generate_sample(float *image, int *label) {
    // Clear the image
    for (int i = 0; i < INPUT_SIZE; i++) {
        image[i] = 0.0f;
    }
    
    // Randomly select a quadrant (0-3)
    *label = rand() % 4;
    
    // Calculate quadrant boundaries
    int half_size = IMG_SIZE / 2;
    int row_start = (*label / 2) * half_size;
    int col_start = (*label % 2) * half_size;
    
    // Place a bright pixel in a random location within the quadrant
    int row = row_start + rand() % half_size;
    int col = col_start + rand() % half_size;
    image[row * IMG_SIZE + col] = 1.0f;
}

// Function to convert label to one-hot encoded vector
void label_to_onehot(int label, float *onehot) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        onehot[i] = (i == label) ? 1.0f : 0.0f;
    }
}

// Function to print the image (for debugging)
void print_image(float *image) {
    for (int i = 0; i < IMG_SIZE; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
            printf("%c", image[i * IMG_SIZE + j] > 0.5f ? 'X' : '.');
        }
        printf("\n");
    }
}

// Function to initialize network parameters with random values
void initialize_parameters() {
    // Initialize weights with small random values
    int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
    for (int l = 0; l < num_layers - 1; l++) {
        int input_size = layer_sizes[l];
        int output_size = layer_sizes[l+1];
        int offset = weight_offsets[l];
        
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                weights[offset + i * input_size + j] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
            }
        }
    }
    
    // Initialize biases to zero
    for (int i = 0; i < activation_offsets[num_layers-1]; i++) {
        biases[i] = 0.0f;
    }
}

int main() {
    srand(time(NULL)); // Seed random number generator
    
    int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
    
    // Initialize network buffers and offsets
    initialize_network(layer_sizes, num_layers, &weights, &biases, &Z_values, &A_values,
                      &dZ_values, &dW, &db, &dA, &weight_offsets, &activation_offsets);
    
    // Initialize parameters
    initialize_parameters();
    
    float learning_rate = 0.1f;
    
    // Training loop
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        printf("\n\n================ Epoch %d ================\n", epoch + 1);
        float epoch_loss = 0.0f;
        int correct = 0;
        
        for (int s = 0; s < NUM_SAMPLES; ++s) {
            // Generate sample
            float sample[INPUT_SIZE];
            int label;
            generate_sample(sample, &label);
            
            // Convert label to one-hot encoding
            float target[OUTPUT_SIZE];
            label_to_onehot(label, target);
            
            // Forward pass
            forward_pass(sample, weights, biases, Z_values, A_values, 
                        layer_sizes, num_layers, weight_offsets, activation_offsets);
            
            // Get output and predicted class
            float *output = &A_values[activation_offsets[num_layers - 1]];
            int predicted = 0;
            float max_val = output[0];
            for (int i = 1; i < OUTPUT_SIZE; i++) {
                if (output[i] > max_val) {
                    max_val = output[i];
                    predicted = i;
                }
            }
            
            // Calculate loss (cross-entropy)
            float loss = -logf(output[label] + 1e-10f);
            epoch_loss += loss;
            
            // Count correct predictions
            if (predicted == label) {
                correct++;
            }
            
            // Backward pass
            backward_pass(target, weights, biases, Z_values, A_values, 
                         dZ_values, dW, db, dA, layer_sizes, num_layers, 
                         weight_offsets, activation_offsets);
            
            // Update weights and biases
            update_weights(weights, biases, dZ_values, A_values, layer_sizes,
                          num_layers, weight_offsets, activation_offsets,
                          learning_rate);
        }
        
        // Print epoch statistics
        printf("Average loss: %.4f\n", epoch_loss / NUM_SAMPLES);
        printf("Accuracy: %.2f%%\n", (float)correct / NUM_SAMPLES * 100.0f);
        
        // Test with a few examples
        printf("\nTest examples:\n");
        for (int t = 0; t < 4; t++) {
            float test_sample[INPUT_SIZE];
            int test_label = t; // Test one sample from each quadrant
            generate_sample(test_sample, &test_label);
            
            forward_pass(test_sample, weights, biases, Z_values, A_values, 
                        layer_sizes, num_layers, weight_offsets, activation_offsets);
            
            float *output = &A_values[activation_offsets[num_layers - 1]];
            int predicted = 0;
            float max_val = output[0];
            for (int i = 1; i < OUTPUT_SIZE; i++) {
                if (output[i] > max_val) {
                    max_val = output[i];
                    predicted = i;
                }
            }
            
            printf("Sample %d (quadrant %d):\n", t+1, test_label+1);
            print_image(test_sample);
            printf("Predicted: %d (confidence: %.2f%%)\n", predicted+1, max_val*100);
            printf("\n");
        }
    }
    
    // Free allocated memory
    free(weights);
    free(biases);
    free(Z_values);
    free(A_values);
    free(dZ_values);
    free(dW);
    free(db);
    free(dA);
    free(weight_offsets);
    free(activation_offsets);
    
    return 0;
}
