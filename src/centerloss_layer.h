#ifndef CENTERLOSS_LAYER_H
#define CENTERLOSS_LAYER_H
#include "layer.h"
#include "network.h"

typedef layer centerloss_layer;

void centerloss_array(float *input, int n, float temp, float *output);
centerloss_layer make_centerloss_layer(int batch, int inputs, int outputs, int groups);
void forward_centerloss_layer(const centerloss_layer l, network net);
void backward_centerloss_layer(const centerloss_layer l, network net);
void update_centerloss_layer(layer l, update_args a);

#ifdef GPU
void pull_centerloss_layer_output(const centerloss_layer l);
void forward_centerloss_layer_gpu(const centerloss_layer l, network net);
void backward_centerloss_layer_gpu(const centerloss_layer l, network net);
#endif

#endif
