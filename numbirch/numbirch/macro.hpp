/**
 * @file
 */
#pragma once

#ifdef __CUDACC__
#define NUMBIRCH_HOST __host__
#else
#define NUMBIRCH_HOST
#endif

#ifdef __CUDACC__
#define NUMBIRCH_DEVICE __device__
#else
#define NUMBIRCH_DEVICE
#endif

#ifdef __CUDACC__
#define NUMBIRCH_HOST_DEVICE __host__ __device__
#else
#define NUMBIRCH_HOST_DEVICE
#endif

/**
 * @internal
 * 
 * @def ARRAY
 * 
 * Constructs the type `Array<T,D>`.
 */
#define ARRAY(T, D) Array<T,D>

/**
 * @internal
 * 
 * @def NUMBIRCH_REAL
 * 
 * Macro to set the default floating point type. Valid values are `float` and
 * `double`.
 */
#ifndef NUMBIRCH_REAL
#define NUMBIRCH_REAL double
#endif
