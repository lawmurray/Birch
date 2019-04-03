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
 * Was `--enable-clone-memo` used for this build?
 */
function configCloneMemo() -> Boolean {
  cpp{{
  return ENABLE_CLONE_MEMO;
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
 * The value of `--clone-memo-delta` used when building.
 */
function configCloneMemoDelta() -> Integer {
  cpp{{
  return CLONE_MEMO_DELTA;
  }}
}

/**
 * Output all config options to a buffer.
 */
function configWrite(buffer:Buffer) {
  buffer.set("memory-pool", configMemoryPool());
  buffer.set("lazy-deep-clone", configLazyDeepClone());
  buffer.set("clone-memo", configCloneMemo());
  buffer.set("clone-memo-initial-size", configCloneMemoInitialSize());
  buffer.set("clone-memo-delta", configCloneMemoDelta());
}
