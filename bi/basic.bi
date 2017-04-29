/**
 * Built-in types
 * --------------
 */
type Boolean;
type Real64;
type Real32;
type Integer64;
type Integer32;
type Real = Real64;
type Integer = Integer64;
type String;

/**
 * Conversions
 * -----------
 */
function Real64(x:Real32) -> Real64 {
  cpp{{
  return x;
  }}
}

function Real32(x:Real64) -> Real32 {
  cpp{{
  return static_cast<float>(x);
  }}
}

function Integer64(x:Integer32) -> Integer64 {
  cpp{{
  return x;
  }}
}

function Integer32(x:Integer64) -> Integer32 {
  cpp{{
  return static_cast<int32_t>(x);
  }}
}
