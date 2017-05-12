#ifndef _FILTER_H
#define _FILTER_H

const float pi = 3.14159;

struct filter
{
    int k;
    float *h;
};

struct linear_filter
{
    int k;
    float *hx;
    float *hy;
};

struct filter_bank
{
    int n;
    filter *bank;
};

float *get_gaussian_kernel(float sigma, int k_size);

void dump_filter(float *filter, int k);

void create_DoG_bank (filter_bank *fb, int n, float k, float sigma0, float levels[]);

void dump_bank(filter_bank *fb);

#endif
