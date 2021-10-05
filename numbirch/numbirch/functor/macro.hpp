/**
 * @file
 */
#pragma once

#ifdef BACKEND_CUDA
#define DEVICE __device__
#else
#define DEVICE
#endif

#define PI 3.1415926535897932384626433832795
