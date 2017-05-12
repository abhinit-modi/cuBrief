#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include "filter.h"
#include <math.h>

using namespace std;

float *get_gaussian_kernel(float sigma, int k_size)
{
    int radius = k_size / 2;
    float k = 1/(2 * pi * sigma * sigma);

    /* Allocate memory for the filter */
    float *filter = (float *) malloc(k_size*k_size*sizeof(float));

    float den = 2 * (sigma * sigma);

    float sum = 0.0;

    for (int y = radius*-1; y <= radius; y++)
    {
        for (int x = radius*-1; x <= radius; x++)
        {
            float num = (x*x + y*y);
            float res = k * exp((-1*num)/den);
            int access = k_size * (y + radius) + (x + radius) ;
            filter[k_size * (y + radius) + (x + radius) ] = res;
            sum+= res;
        }
    }

    for (int y = radius*-1; y <= radius; y++)
        for (int x = radius*-1; x <= radius; x++)
            if (sum != 0.0) 
            filter[k_size * (y + radius) + (x + radius)]/= sum;

    return filter;
}

void dump_filter(float *filter, int k)
{
    for (int i = 0; i < k; i++)
    {
        for(int j = 0; j < k; j++)
            cout<<filter[k*i + j]<<" ";
        cout<<endl;
    }

}

void create_DoG_bank (filter_bank *fb, int n, float k, float sigma0, float levels[])
{
    /* Allocate memory for the filter bank */
    fb->n = n;
    fb->bank = (filter *) malloc(sizeof(filter) * n);
    /* Add a small delta to the argument of floor.
       floor(24.0f) gives 23.0.
    */
    float delta = 0.001;
 
    /* Create each filter using the specifications in the parameters */
    
    for (int i = 0; i < n; i++)
    {
        float sigma = sigma0 * pow(k, levels[i]);
        int ksize = floor(6*sigma + delta) + 1;
        //cout << "sigma is "<<sigma<<" and ksize is "<< ksize << endl;
        fb->bank[i].h = get_gaussian_kernel(sigma, ksize);
        fb->bank[i].k = ksize; 
    }
}

void dump_bank(filter_bank *fb)
{
    for (int i = 0; i < fb->n; i++)
    {
        cout<<"Dumping filter at level "<<i<<endl;
        dump_filter(fb->bank[i].h, fb->bank[i].k);
    }
}


