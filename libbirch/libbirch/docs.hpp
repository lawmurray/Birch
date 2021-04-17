/**
 * @mainpage
 *
 * LibBirch is a C++ template library intended as a compile target for
 * probabilistic programming languages (PPLs). It provides basic types,
 * [Eigen](https://eigen.tuxfamily.org/) integration, and dynamic memory
 * management (reference counting, cycle collection, lazy deep copy). This is
 * particularly important for memory efficiency in PPLs using Sequential Monte
 * Carlo (SMC) and related methods for inference.
 *
 * LibBirch was developed primarily to support the [Birch](https://birch.sh)
 * probabilistic programming language, but may be useful for other purposes
 * too.
 *
 * ## Reference counting
 *
 * LibBirch provides reference counting support in combination with lazy deep
 * copy operations, as described in @ref Murray2020 "Murray (2020)". To use
 * this functionality, a class must derive from Any.
 *
 * Reference-counted objects in LibBirch use three counts, rather than
 * the usual two (shared and weak), in order to support these lazy deep copy
 * operations. The counts are:
 *
 *   - a *shared* count,
 *   - a *weak* count, and
 *   - a *memo* count.
 *
 * The shared and weak counts behave as normal. The memo count is used for
 * keys in the memos used to bookkeep lazy deep copy operations.
 *
 * The movement of the three counts triggers the following operations:
 *
 *   -# When the shared count reaches zero, the object is *destroyed*.
 *   -# If the weak and memo weak counts reach 0, the object is *deallocated*.
 *
 * Shared and Weak pointers determine which objects are reachable from the
 * user program. Memo uses the third type internally for keys: it surrenders
 * these during its own cleanup operations if no other shared or weak pointers
 * exist to an object.
 *
 * ## Cycle collection
 *
 * Shared and Weak pointers are can be insufficient, and user-programmed logic
 * to tear down data structures can be problematic with lazy deep copy
 * operations: it can end up creating new objects for the purpose of
 * destroying them. Furthermore, Memo can induce reference cycles that would
 * not otherwise exist in the user program.
 *
 * For this reason, LibBirch provides the cycle collection algorithm after
 * @ref Bacon2001 "Bacon & Rajan (2001)", with some minor adaptations.
 *
 * ## References
 *
 * @anchor Murray2020
 * L.M. Murray (2020). [Lazy object copy as a platform for
 * population-based probabilistic programming](https://arxiv.org/abs/2001.05293).
 *
 * @anchor Bacon2001
 * D.F. Bacon and V.T. Rajan (2001). [Concurrent Cycle Collection in
 * Reference Counted Systems](https://dx.doi.org/10.1007/3-540-45337-7_12).
 * *ECOOP 2001 --- Object-Oriented Programming*. 207--235.
 */
