/**
 * @file
 */
#pragma once

#ifdef __CUDACC__
#define HOST __host__
#else
#define HOST
#endif

#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif

#ifdef __GNUG__
#define PURE __attribute__((const))
#else
#define PURE
#endif

#define PI 3.1415926535897932384626433832795
