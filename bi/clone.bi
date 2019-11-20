/**
 * Deep clone an object or fiber.
 */
function clone<Type>(o:Type) -> Type {
  cpp{{
  return o.clone(context_);
  }}
}

/**
 * Deep clone an object or fiber multiple times to construct an array.
 *
 * - o: Source object.
 * - length: Length of vector.
 */
function clone<Type>(o:Type, length:Integer) -> Type[_] {
  auto f <- @() -> Type { return clone<Type>(o); };
  cpp{{
  return libbirch::make_array<Type>(context_, libbirch::make_shape(length), f);
  }}
}
