/**
 * Get the amount of memory (in bytes) currently in use. If the pooled memory
 * allocator is enabled, this includes memory currently held in pools but
 * not in use, and will never decrease in time.
 */
function memoryUse() -> Integer {
  cpp{{
  return libbirch::memoryUse();
  }}
}
