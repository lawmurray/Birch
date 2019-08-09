/**
 * Get the amount of memory currently allocated on the heap, in bytes.
 */
function memoryUse() -> Integer {
  cpp{{
  return libbirch::memoryUse.load();
  }}
}
