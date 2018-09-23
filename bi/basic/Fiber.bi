/**
 * Deep clone a fiber.
 */
function clone(f:Variate!) -> Variate! {
  cpp{{
  return f_.clone();
  }}
}
