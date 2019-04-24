/**
 * Deep clone an object or fiber.
 *
 * If code is built with `--enable-lazy-deep-clone` (the default), this
 * initializes a lazy deep clone of the object or fiber, such that any other
 * objects or fibers reachable through it are only copied when necessary, and
 * may never be copied at all. If code is built with
 * `--disable-lazy-deep-clone` then all objects are copied immediately.
 *
 * For objects used with a lazy deep clone, consider using persistent data
 * structures such as List and Queue to maximise sharing and memory
 * efficiency.
 *
 * !!! bug
 *     The lazy deep clone mechanism is still being developed. It has bugs
 *     for some use cases. In particular, once an object is passed as an
 *     argument to `clone<Type>()`, avoid using that object again.
 */
function clone<Type>(o:Type) -> Type {
  cpp{{
  return o.clone();
  }}
}
