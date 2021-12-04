/**
 * @file
 * 
 * @defgroup numeric Numeric
 * Asynchronous numerics.
 * 
 * @defgroup unary Unary transformations
 * @ingroup numeric
 * Batched unary transformations of standard and special math functions and
 * gradients, such as sin() and digamma().
 * 
 * @defgroup binary Binary transformations
 * @ingroup numeric
 * Batched binary transformations of standard and special math functions and
 * gradients, such as pow() and lbeta().
 * 
 * @defgroup ternary Ternary transformations
 * @ingroup numeric
 * Batched ternary transformations of standard and special math functions and
 * gradients, such as ibeta().
 * 
 * @defgroup reduce Reductions
 * @ingroup numeric
 * Reductions such as sum() and count().
 * 
 * @defgroup la Linear algebra
 * @ingroup numeric
 * Linear algebra operations, such as matrix multiplication, solve() and
 * cholsolve().
 * 
 * @defgroup random Pseudorandom number generation
 * @ingroup numeric
 * Batched pseudorandom number generation, as well as multivariate and matrix
 * distributions.
 * 
 * @defgroup array Array
 * Multidimensional arrays with copy-on-write and unified memory management.
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
#include "numbirch/unary.hpp"
#include "numbirch/binary.hpp"
#include "numbirch/ternary.hpp"
#include "numbirch/reduce.hpp"
#include "numbirch/random.hpp"
