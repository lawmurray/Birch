/**
 * @file
 */
#pragma once

#ifdef BACKEND_CUDA
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif
