/**
 * Deep clone an object or fiber.
 */
function clone<Type>(o:Type) -> Type {
  cpp{{
  ///@todo
  //return o->clone();
  }}
}

/**
 * Deep clone an object or fiber multiple times to construct an array.
 *
 * - o: Source object.
 * - length: Length of vector.
 */
function clone<Type>(o:Type, length:Integer) -> Type[_] {
  auto l <- @(n:Integer) -> Type { return clone<Type>(o); };
  cpp{{
  return libbirch::make_array_from_lambda<Type>(libbirch::make_shape(length), l);
  }}
}
