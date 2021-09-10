#include "ann_def.h"

typedef struct ANN_Neuron {
  float *weights;
  float bias;
  float input;
  float output;   // activate(input)
  float dError;   // dErrorTotal/dOutput
  float alpha;    // dErrorTotal/dInput
} ANN_Neuron;

typedef struct ANN_Layer {
  uint8_t numOfNeuron;
  ANN_Neuron *neurons;
} ANN_Layer;


struct ANN_Config {
  float learningRate;
  uint8_t numOfInput;
  uint8_t numOfhiddenLayer;
  uint8_t *hiddenLayerLengths;
  uint8_t numOfOutput;
  uint8_t numOfLayers;
};

typedef struct ANN_Handler {
  struct ANN_Config config;
  ANN_Layer *layers;
  ANN_Layer *inputLayer;
  ANN_Layer *outputLayer;
  float error;
} ANN_HandlerTypedef;

struct ANN_TrainData {
  float *inputs;
  float *outputs;
};

uint8_t ANN_IsLog;

void ANN_Init(ANN_HandlerTypedef *hann, struct ANN_Config config);
void ANN_DeInit(ANN_HandlerTypedef *hann);
void ANN_Run(ANN_HandlerTypedef *hann, float *inputs, float *outputs);
void ANN_Learn(ANN_HandlerTypedef *hann, struct ANN_TrainData *data, int len, int iter);
void ANN_PrintNet(ANN_HandlerTypedef *hann);
void ANN_PrintW(ANN_HandlerTypedef *hann);