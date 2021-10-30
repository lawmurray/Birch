/**
 * @file
 */
#pragma once

#ifdef BACKEND_CUDA
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

#define PI 3.1415926535897932384626433832795
