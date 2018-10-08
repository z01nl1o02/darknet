#include "centerloss_layer.h"
#include "softmax_layer.h"
#include "connected_layer.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <memory.h>



centerloss_layer make_centerloss_layer(int batch, int inputs, int outputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "centerloss                                       %4d\n",  inputs);
    centerloss_layer l = {0};

    layer tmp_layer = make_connected_layer(batch,inputs,outputs,LEAKY, 0, 1);
    l.cl_fc_layer = calloc(1,sizeof(layer));
    memcpy(l.cl_fc_layer,&tmp_layer,sizeof(layer));

    tmp_layer = make_softmax_layer(batch, outputs, groups);
    l.cl_softmax_layer = calloc(1,sizeof(layer));
    memcpy(l.cl_softmax_layer,&tmp_layer,sizeof(layer));

    l.type = CENTERLOSS;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = outputs;
    l.cl_centers = calloc(inputs * outputs, sizeof(float));
    l.cl_centers_delta = calloc(outputs * inputs, sizeof(float));

    fill_cpu(l.inputs * l.outputs, 0, l.cl_centers, 1);
    fill_cpu(l.inputs * l.outputs, 0, l.cl_centers_delta, 1);

    l.delta = calloc(inputs*batch, sizeof(float));
    l.loss = calloc(inputs*batch, sizeof(float));
    l.output = calloc(outputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.alpha = 0.5; //center update momentum
    l.lambda = 0.01; //Ls + 0.5*lambda * Lc

    l.forward = forward_centerloss_layer;
    l.backward = backward_centerloss_layer;
    l.update = update_centerloss_layer;
    #ifdef GPU
    l.forward_gpu = forward_centerloss_layer_gpu;
    l.backward_gpu = backward_centerloss_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
    #endif
    return l;
}

void centerloss_get_loss(float* input, int batch, int inputs, int outputs, float* truth,float* centers, float* center_delta, float* delta, float* loss)
{
    int batchidx;
    //not memset center_delta here. it is the respondence of caller
    for(batchidx = 0; batchidx < batch; batchidx++ )
    {
        float* current_input = input + batchidx * inputs;
        float* current_truth = truth + batchidx * inputs;
        float* current_delta = delta + batchidx * inputs;
        float* current_loss = loss + batchidx * inputs;
        int cid = 0;
        while(cid < outputs)
        {
            if(current_truth[cid] > 0.5) break;
            cid++;
        }
        float* current_center = centers + cid * inputs;
        float* current_center_delta = center_delta + cid * inputs;
        int inputidx;
        for(inputidx = 0; inputidx < inputs; inputidx++)
        {
            float tmp = current_input[inputidx] - current_center[inputidx];
            current_loss[inputidx] = tmp * tmp;
            current_delta[inputidx] =  tmp;
            current_center_delta[inputidx] = -tmp;
        }
    }
    return;
}

void forward_centerloss_layer(const centerloss_layer l, network net)
{
    l.cl_fc_layer->forward(*(l.cl_fc_layer), net);
    if(net.train && net.truth)
    {//calculate center_delta
        centerloss_get_loss(net.input,l.batch, l.inputs, l.outputs, net.truth, l.cl_centers, l.cl_centers_delta, l.delta,l.loss);
    }
    l.cl_softmax_layer->forward(*(l.cl_softmax_layer),net);
    if(net.train && net.truth)
    {
        l.cost[0] = l.cl_softmax_layer->loss[0] +  0.5 * l.lambda * sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_centerloss_layer(const centerloss_layer l, network net)
{
    l.cl_softmax_layer->backward(*(l.cl_softmax_layer),net);
    axpy_cpu(l.inputs * l.batch, l.lambda, l.delta, 1, net.delta, 1); 
    l.cl_fc_layer->backward(*(l.cl_fc_layer), net);
}

void update_centerloss_layer(layer l, update_args a)
{
    //update class centers
    axpy_cpu(l.inputs * l.outputs, -l.alpha, l.cl_centers_delta, 1, l.cl_centers,1);
    fill_cpu(l.inputs * l.outputs, 0, l.cl_centers_delta, 1);
    //softmax layer no update
    l.cl_fc_layer->update(*(l.cl_fc_layer),a); 
    return;
}


#ifdef GPU

void pull_centerloss_layer_output(const centerloss_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
}

void forward_centerloss_layer_gpu(const centerloss_layer l, network net)
{
    if(l.softmax_tree){
        softmax_tree(net.input_gpu, 1, l.batch, l.inputs, l.temperature, l.output_gpu, *l.softmax_tree);
        /*
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_gpu(net.input_gpu + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output_gpu + count);
            count += group_size;
        }
        */
    } else {
        if(l.spatial){
            softmax_gpu(net.input_gpu, l.c, l.batch*l.c, l.inputs/l.c, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu);
        }else{
            softmax_gpu(net.input_gpu, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output_gpu);
        }
    }
    if(net.truth && !l.noloss){
        softmax_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
        if(l.softmax_tree){
            mask_gpu(l.batch*l.inputs, l.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
            mask_gpu(l.batch*l.inputs, l.loss_gpu, SECRET_NUM, net.truth_gpu, 0);
        }
        cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_centerloss_layer_gpu(const centerloss_layer layer, network net)
{
    axpy_gpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, net.delta_gpu, 1);
}

#endif
