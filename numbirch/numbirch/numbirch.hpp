/**
 * @file
 * 
 * @defgroup array Array
 * Multidimensional arrays with copy-on-write and unified memory management.
 * 
 * @defgroup array_grad Gradients
 * @ingroup array
 * Gradients of array manipulation functions.
 * 
 * @defgroup transform Transformations
 * Transformations of standard and special math functions such as sin() and
 * lgamma().
 * 
 * @defgroup transform_grad Gradients
 * @ingroup transform
 * Gradients of transformations.
 * 
 * @defgroup reduce Reductions
 * Reductions such as sum() and count().
 * 
 * @defgroup reduce_grad Gradients
 * @ingroup reduce
 * Gradients of reductions.
 * 
 * @defgroup linalg Linear algebra
 * Linear algebra functions, such as matrix multiplication and solve().
 * 
 * @defgroup linalg_grad Gradients
 * @ingroup linalg
 * Gradients of linear algebra functions.
 * 
 * @defgroup random Random number generation
 * Batched pseudorandom number generation.
 *
 * @defgroup memory Memory
 * Asynchronous unified memory management.
 * 
 * @defgroup trait Type traits
 * Type traits used for SFINAE.
 */
#pragma once

#include "numbirch/memory.hpp"
#include "numbirch/numeric.hpp"
#include "numbirch/array.hpp"
#include "numbirch/transform.hpp"
#include "numbirch/reduce.hpp"
#include "numbirch/random.hpp"
