#include <stdio.h>
#include "ANN/ann_def.h"
#include "ANN/ann.h"
#include <stdlib.h>

int main(){
    printf("start\n"); 

    for (int i = 0; i < 10; i++)
    {
        float i = (float)rand()/RAND_MAX;
        i = (2*i) - 1;
        printf("%.4f\n", i);
    }
    

    // return 0;
    struct ANN_TrainData trainData[4] = {
        {(float[]){0.0, 0.0}, (float[]){0}},
        {(float[]){0.0, -1.0}, (float[]){1}},
        {(float[]){1.0, 0.0}, (float[]){1}},
        {(float[]){1.0, -1.0}, (float[]){1}},
    };
    ANN_HandlerTypedef myModel;
    int numOfhiddenLayer = 3;

    struct ANN_Config myConfig = {
        .learningRate = 0.1,
        .numOfInput = 2,
        .numOfhiddenLayer = 3,
        .hiddenLayerLengths = (uint8_t[]){50, 100, 100},
        .numOfOutput = 1,
    };

    printf("init\n"); 
    ANN_Init(&myModel, myConfig);
    float inputs[2] = {1.0, -1.0};
    float outputs[1] = {0.0};
    ANN_Learn(&myModel, trainData, 4, 10000);

    outputs[0] = 0.0;
    printf("Input : %.2f %.2f\n", trainData[0].inputs[0], trainData[0].inputs[1]);
    ANN_Run(&myModel, (float *) trainData[0].inputs, outputs);
    printf("Output : %.4f\n", outputs[0]);

    outputs[0] = 0.0;
    printf("Input : %.2f %.2f\n", trainData[1].inputs[0], trainData[1].inputs[1]);
    ANN_Run(&myModel, (float *) trainData[1].inputs, outputs);
    printf("Output : %.4f\n", outputs[0]);

    outputs[0] = 0.0;
    printf("Input : %.2f %.2f\n", trainData[2].inputs[0], trainData[2].inputs[1]);
    ANN_Run(&myModel, (float *) trainData[2].inputs, outputs);
    printf("Output : %.4f\n", outputs[0]);

    outputs[0] = 0.0;
    printf("Input : %.2f %.2f\n", trainData[3].inputs[0], trainData[3].inputs[1]);
    ANN_Run(&myModel, (float *) trainData[3].inputs, outputs);
    printf("Output : %.4f\n", outputs[0]);
    ANN_DeInit(&myModel);
    return 0;
}