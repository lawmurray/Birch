/**
 * Deep clone an object or fiber.
 *
 * If code is built with `--enable-lazy-deep-clone` (the default), this
 * initializes a lazy deep clone of the object or fiber, such that any other
 * objects or fibers reachable through it are only copied when necessary, and
 * may never be copied at all. If code is built with
 * `--disable-lazy-deep-clone` then all objects are copied immediately.
 *
 * For objects used with a lazy deep clone, consider using recursive data
 * structures such as List and Stack to maximise sharing and memory
 * efficiency.
 */
function clone<Type>(o:Type) -> Type {
  cpp{{
  return o_.clone();
  }}
}
