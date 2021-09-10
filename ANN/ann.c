#include "ann_def.h"
#include "ann.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <unistd.h>

static int debug(const char *c, ...);
static void activate(ANN_Neuron *neu);
static float dActivate(ANN_Neuron *neu);
static void forward(ANN_HandlerTypedef *hann, float * inputs);
static void backwardPropagateErr(ANN_HandlerTypedef *hann, float * expectOutputs, uint8_t isDebug);
static void updateWeight(ANN_HandlerTypedef *hann);

void ANN_Init(ANN_HandlerTypedef *hann, struct ANN_Config config)
{
  uint8_t i, j, k;
  uint8_t prevLayerLength = 0;
  
  config.numOfLayers = config.numOfhiddenLayer + 2;
  hann->config = config;
  hann->layers = malloc(sizeof(ANN_Layer)*config.numOfLayers);

  for (i = 0; i < config.numOfLayers; i++){
    ANN_Layer tmpLayer;
    uint8_t layerLength;
    float addX;
    float addY;
    float currentAddX = -1;
    float currentAddY = -1;

    if (i == 0)
      layerLength = config.numOfInput;
    else if (i < config.numOfLayers-1)
      layerLength = config.hiddenLayerLengths[i];
    else
      layerLength = config.numOfOutput;
    
    if(prevLayerLength > 1) {
      addX = 1.0/((prevLayerLength));
      // printf("2 / (%d*%d) - add w %.4f\n", (int)layerLength, (int)prevLayerLength, addW);
    }
    if(layerLength > 1) {
      addY = 1.0/((layerLength-1));
      // printf("2 / (%d*%d) - add w %.4f\n", (int)layerLength, (int)prevLayerLength, addW);
    }
    
    tmpLayer.numOfNeuron = layerLength;
    tmpLayer.neurons = malloc(sizeof(ANN_Neuron)*layerLength);

    for (j = 0; j < layerLength; j++){
      ANN_Neuron tmpNeuron;

      tmpNeuron.bias = 0;
      if(prevLayerLength){
        tmpNeuron.weights = malloc(sizeof(float)*prevLayerLength);
        currentAddY = currentAddX;
        currentAddX += addX;
        if(currentAddX > 1) currentAddX -= 2;
        for (k = 0; k < prevLayerLength; k++){
          tmpNeuron.weights[k] = currentAddY;
          currentAddY += addY;
          if(currentAddY > 1) currentAddY -= 2;
        }
      }
      else
        tmpNeuron.weights = 0;
      tmpLayer.neurons[j] = tmpNeuron;
    }
    hann->layers[i] = tmpLayer;
    prevLayerLength = layerLength;
  }

  hann->inputLayer = &(hann->layers[0]);
  hann->outputLayer = &(hann->layers[config.numOfLayers - 1]);
}

void ANN_DeInit(ANN_HandlerTypedef *hann)
{
  uint8_t i, j;
  for (i = 0; i < hann->config.numOfLayers; i++){
    for (j = 0; j < hann->layers[i].numOfNeuron; j++){
      // if(hann->layers[i].neurons[j].weights)
      //   free(hann->layers[i].neurons[j].weights);
    }
    free(hann->layers[i].neurons);
  }
  free(hann->layers);
}

void ANN_Run(ANN_HandlerTypedef *hann, float *inputs, float *outputs)
{
  uint8_t i;
  ANN_Layer *outputLayer = hann->outputLayer;
  
  forward(hann, inputs);
  if(outputs){
    for (i = 0; i < outputLayer->numOfNeuron; i++){
      outputs[i] = outputLayer->neurons[i].output;
    }
  }
}

void ANN_Learn(ANN_HandlerTypedef *hann, struct ANN_TrainData *data, int len, int iter)
{
  int i, j, k;
  uint8_t isDebug = 0;

  for (i = 0; i < iter; i++){
    // printf("iter #%d\n", i);
    if(i == (iter-1)) isDebug = 1;
    for (j = 0; j < len; j++){
      forward(hann, data[j].inputs);
      backwardPropagateErr(hann, data[j].outputs, isDebug);
      updateWeight(hann);
    }
  }
}

void ANN_PrintNet(ANN_HandlerTypedef *hann)
{
  printf("input    : %d\n", hann->config.numOfInput);
  printf("h_layer  : ");
  for (uint8_t i = 0; i < hann->config.numOfhiddenLayer; i++){
    printf("-%d", hann->layers[i].numOfNeuron);
  }
  
  printf("\noutput   : %d\n", hann->config.numOfOutput);
}

void ANN_PrintW(ANN_HandlerTypedef *hann)
{
  uint8_t i, j, k;
  int prevLayerN = hann->layers[0].numOfNeuron;

  for (i = 1; i < hann->config.numOfLayers; i++){
    printf("layer #%d\n", i);
    for (j = 0; j < hann->layers[i].numOfNeuron; j++){
      printf("\t");
      for (k = 0; k < prevLayerN; k++){
        float w = hann->layers[i].neurons[j].weights[k];
        printf("%.4f ", w);
      }
      printf("\n");
    }
    prevLayerN = hann->layers[i].numOfNeuron;
  }
  
}

static void activate(ANN_Neuron *neu)
{
  neu->output = 1.0 / (1.0 + (float) exp((double)(-(neu->input))));
}

static float dActivate(ANN_Neuron *neu)
{
  // neu->output is activation func result
  // a'(x) = a(x) * (1 - a(x))
  return neu->output * (1.0 - neu->output);
}

static void forward(ANN_HandlerTypedef *hann, float * inputs)
{
  uint8_t i, j, k;
  ANN_Layer *prevLayer = &(hann->layers[0]);
  
  for (i = 0; i < prevLayer->numOfNeuron; i++){
    prevLayer->neurons[i].output = inputs[i];
  }
  
  for (i = 1; i < hann->config.numOfLayers; i++){
    for (j = 0; j < hann->layers[i].numOfNeuron; j++){
      ANN_Neuron *neuron = &(hann->layers[i].neurons[j]);

      neuron->input = neuron->bias;
      for (k = 0; k < prevLayer->numOfNeuron; k++){
        neuron->input += neuron->weights[k] * prevLayer->neurons[k].output;
      }
      activate(neuron);
    }
    prevLayer = &(hann->layers[i]);
  }
}

static void backwardPropagateErr(ANN_HandlerTypedef *hann, float *expectOutputs, uint8_t isDebug)
{
  uint8_t i, j, k;
  ANN_Layer *frontLayer = NULL;

  // sleep(1);
  hann->error = 0;
  for (i = hann->config.numOfLayers-1; i > 0; i--){
    for (j = 0; j < hann->layers[i].numOfNeuron; j++){
      ANN_Neuron *neuron = &(hann->layers[i].neurons[j]);
      float error = 0.0;

      if(frontLayer == NULL){
        neuron->dError = neuron->output - expectOutputs[j];
        hann->error += neuron->dError;
        if(isDebug){
          printf("Error [#%d]: %.4f\n", j, neuron->dError);
          printf("\tExpect [#%d]: %.4f\n", j, expectOutputs[j]);
          printf("\tGet [#%d]: %.4f\n", j, neuron->output);
        }
      } else {
        for (k = 0; k < frontLayer->numOfNeuron; k++){
          neuron->dError = frontLayer->neurons[k].alpha * frontLayer->neurons[k].weights[j];
        }
      }
      
      //generate alpha
      neuron->alpha = neuron->dError * dActivate(neuron);
    }
    frontLayer = &(hann->layers[i]);
  }
}

static void updateWeight(ANN_HandlerTypedef *hann)
{
  uint8_t i, j, k;
  ANN_Layer *prevLayer = &(hann->layers[0]);
    
  for (i = 1; i < hann->config.numOfLayers; i++){
    for (j = 0; j < hann->layers[i].numOfNeuron; j++){
      ANN_Neuron *neuron = &(hann->layers[i].neurons[j]);
      for (k = 0; k < prevLayer->numOfNeuron; k++){
        neuron->weights[k] -= hann->config.learningRate * neuron->alpha * prevLayer->neurons[k].output;
      }
      neuron->bias -= hann->config.learningRate * neuron->alpha;
    }
    prevLayer = &(hann->layers[i]);
  }
}
