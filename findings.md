

## Findings:

#### Serial operation

**Total time:** 398.08 seconds
**Average Samples per Second:** 150 sps 

```
Random Test Image (True Label: 6):
               *#
              .@@
              *@@
              @@
             *@*
             @@
            #@#
            @@.
           .@@..
           *@@@@@
           *@@@@@@.
           @@@*.@@@
           @@.   @@.
          .@*    @@.
          .@.   .@@
          .@    @@.
           @   *@@
           @@.#@@*
           #@@@@#
           .@@@*

starting training
Epoch 1, Accuracy: 96.05%, Avg Loss: 0.2700, Time: 79.87 seconds
Epoch 2, Accuracy: 96.62%, Avg Loss: 0.1017, Time: 80.72 seconds
Epoch 3, Accuracy: 96.50%, Avg Loss: 0.0630, Time: 78.32 seconds
Epoch 4, Accuracy: 97.70%, Avg Loss: 0.0402, Time: 79.55 seconds
Epoch 5, Accuracy: 97.66%, Avg Loss: 0.0271, Time: 79.62 seconds
```


#### Parallel per Operation

**Total time:** 112.75 seconds
**Average Samples per Second:** 535 sps


Retains the accuracy of the serial implementation. While increasing the speed by around $3.5\times$. Success!

```
Random Test Image (True Label: 4):

       #@
       @@         .@#
       @@        .@@#
      *@@        #@@
      #@@        @@@
      *@@        #@@
       @@        .@*
       @*        .@*
       @#        @@*
       @@*@@#..@.@@*
       ..*@@@@@@@@@#
          @@@@@@@@@*
           ***   #@*
                 .@*
                 .@*
                 .@*
                 .@*
                 .@*
                 .@*
                  **


starting training
Epoch 1, Accuracy: 95.73%, Avg Loss: 0.2718, Time: 21.48 seconds
Epoch 2, Accuracy: 96.62%, Avg Loss: 0.1017, Time: 21.74 seconds
Epoch 3, Accuracy: 96.50%, Avg Loss: 0.0630, Time: 22.71 seconds
Epoch 4, Accuracy: 97.70%, Avg Loss: 0.0402, Time: 22.64 seconds
Epoch 5, Accuracy: 97.66%, Avg Loss: 0.0271, Time: 24.18 seconds
Predicted Label: 4
```


#### Parallel per Sample

This method had a similar runtime to the **per operation** method ( ~25 seconds/epoch). However, it consistently scored a lower accuracy using the MNIST dataset (maximum of 70% accuracy, with the accuracy being volatile, sometimes increasing or decreasing drastically from epoch to epoch). Altering the parameters of the neural network, such as increasing the `learning rate` or lowering the `batch size`; drastically changes the outcomes, making this a very volatile method on this MNIST usecase. However it still holds potential for very large (wide datasets) with small calculations for each sample (shallow).

**Sample training output**

**Total time:** 127 seconds
**Average Samples per Second:** 468 sps

```
Random Test Image (True Label: 7):

		   .@#.
           #@@@*. *
           @@@@@@@@@
           @@##@@@@@.
          *@.   **@@#
          #@      @@.
         .@@     .@@
          *      @@*
                #@@
               .@@
               *@#
              .@@
             *@@#
             #@@
            @@@
           #@@*
          #@@*
         .@@*
         *@@
          #@

starting training
Epoch 1, Accuracy: 47.19%, Avg Loss: -nan, Time: 24.46 seconds
Epoch 2, Accuracy: 60.61%, Avg Loss: -nan, Time: 25.05 seconds
Epoch 3, Accuracy: 62.90%, Avg Loss: -nan, Time: 24.86 seconds
Epoch 4, Accuracy: 62.96%, Avg Loss: 2.4094, Time: 24.93 seconds
Epoch 5, Accuracy: 66.81%, Avg Loss: 2.3968, Time: 24.97 seconds
Predicted Label: 1
```

