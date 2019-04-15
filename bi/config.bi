/**
 * Was `--enable-memory-pool` used for this build?
 */
function configMemoryPool() -> Boolean {
  cpp{{
  return ENABLE_MEMORY_POOL;
  }}
}

/**
 * Was `--enable-lazy-deep-clone` used for this build?
 */
function configLazyDeepClone() -> Boolean {
  cpp{{
  return ENABLE_LAZY_DEEP_CLONE;
  }}
}

/**
 * The value of `--clone-memo-initial-size` used when building.
 */
function configCloneMemoInitialSize() -> Integer {
  cpp{{
  return CLONE_MEMO_INITIAL_SIZE;
  }}
}

/**
 * Output all config options to a buffer.
 */
function configWrite(buffer:Buffer) {
  buffer.set("memory-pool", configMemoryPool());
  buffer.set("lazy-deep-clone", configLazyDeepClone());
  buffer.set("clone-memo-initial-size", configCloneMemoInitialSize());
}
