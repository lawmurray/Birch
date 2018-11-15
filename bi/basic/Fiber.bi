/**
 * Deep clone a fiber.
 */
function clone<Type>(f:Type) -> Type {
  cpp{{
  return f_.clone();
  }}
}
