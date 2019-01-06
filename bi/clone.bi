/**
 * Deep clone an object or fiber.
 *
 * !!! caution
 *     The intent of this function is to provide lazy deep clone
 *     functionality. The algorithm used is a work in progress, however, and
 *     produces incorrect results in some cases. It produces correct results
 *     for its use cases within the standard library itself, but mileage may
 *     vary outside of this context at this stage.
 */
function clone<Type>(o:Type) -> Type {
  cpp{{
  return o_.clone();
  }}
}
