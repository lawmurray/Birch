/**
 * Deep clone a fiber.
 */
function clone<Type>(f:Type) -> Type {
  cpp{{
  return pop_context(push_context(f_).clone());
  }}
}
