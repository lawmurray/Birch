/**
 * Make an object, with the type given as an argument.
 *
 *   - Type: The type.
 *
 * Returns: An optional with a value if successful, or no value if not
 * successful.
 *
 * The make will not succeed if the type is a class with initialization
 * parameters, or a compound type that includes such a class.
 */
function make<Type>() -> Type? {
  cpp{{
  return libbirch::make<Type>();
  }}
}

/**
 * Make an object, with the class given as a string.
 *
 *   - name: Name of the class.
 *
 * Returns: An optional with a value if successful, or no value if not
 * successful.
 *
 * The make will not succeed if the class has initialization parameters.
 */
function make(name:String) -> Object? {
  result:Object?;
  symbol:String <- "make_" + name + "_";
  cpp{{
  using make_t = bi::type::Object*();
  void* addr = dlsym(RTLD_DEFAULT, symbol.c_str());
  if (addr) {
    result = libbirch::LazySharedPtr<bi::type::Object>(reinterpret_cast<make_t*>(addr)());
  }
  }}
  if !result? {
    warn("could not make object of type " + name +
        "; class may not exist or may require initialization arguments.");
  }
  return result;
}

/**
 * Make an object, with the class possibly given as a string.
 *
 *   - name: Name of the class.
 *
 * Returns: An optional with a value if successful, or no value if not
 * successful or `name` has no value.
 */
function make(name:String?) -> Object? {
  if name? {
    return make(name!);
  } else {
    return nil;
  }
}

/**
 * Make an object, with the class given in a buffer.
 *
 *   - buffer: The buffer.
 *
 * Returns: An optional with a value if successful, or no value if not
 * successful.
 *
 * If the buffer contains a key `class`, an object of that class is 
 * constructed. The buffer is then passed to the `read()` function of the new
 * object.
 *
 * The make will not succeed if the class has initialization parameters.
 */
function make(buffer:Buffer) -> Object? {
  result:Object?;
  auto className <- buffer.getString("class");
  if className? {
    result <- make(className!);
  }
  if result? {
    result!.read(buffer);
  }
  return result;
}

/**
 * Make an object, with the class possibly given in a buffer.
 *
 *   - buffer: The buffer.
 *
 * Returns: An optional with a value if successful, or no value if not
 * successful.
 */
function make(buffer:Buffer?) -> Object? {
  if buffer? {
    return make(buffer!);
  } else {
    return nil;
  }
}
