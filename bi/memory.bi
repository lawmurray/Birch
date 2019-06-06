/**
 * Get the amount of memory (in bytes) currently in use.
 */
function memoryUse() -> Integer {
  cpp{{
  return libbirch::memoryUse;
  }}
}

/**
 * Get the amount of memory (in bytes) currently pooled. If the pooled memory
 * allocator is enabled, this is always greater than or equal to the amount
 * of memory currently in use, and only ever increases in time. If the pooled
 * memory allocator is disabled, this is always equal to the amount of memory
 * currently in use.
 */
function memoryPool() -> Integer {
  cpp{{
  #if ENABLE_MEMORY_POOL
  return libbirch::buffer.load(std::memory_order_relaxed) - libbirch::bufferStart;
  #else
  return libbirch::memoryUse;
  #endif
  }}
}
