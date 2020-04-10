/**
 * Deep clone an object.
 */
function clone<Type>(o:Type) -> Type {
  cpp{{
  return libbirch::clone(o);
  }}
}

/**
 * Deep clone an array.
 *
 * - o: Source array.
 */
function clone<Type>(o:Type[_]) -> Type[_] {
  return transform<Type>(o, @(x:Type) -> Type { return clone(x); });
}

/**
 * Deep clone an object multiple times to construct an array.
 *
 * - o: Source object.
 * - length: Length of vector.
 */
function clone<Type>(o:Type, length:Integer) -> Type[_] {
  /* use a C++ lambda function with reference capture here, rather than a
   * Birch lambda function with copy capture, to avoid incrementing the
   * reference count on `o`; this can have significant performance
   * implications around the single-reference optimization */
  cpp{{
  auto l = [&](int64_t n) { return libbirch::clone(o); };
  return libbirch::make_array_from_lambda<Type>(
      libbirch::make_shape(length), l);
  }}
}
