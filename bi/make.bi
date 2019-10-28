/**
 * Make an object.
 *
 *   - name: Name of the class.
 *
 * Return: if `name` names a class with no initialization parameters,
 * constructs an object of that class and returns it in an optional,
 * otherwise returns an optional with no value.
 */
function make(name:String) -> Object? {
  symbol:String <- "make_" + name + "_";
  cpp{{
  using make_t = bi::type::Object*(libbirch::Label*);
  void* addr = dlsym(RTLD_DEFAULT, symbol.c_str());
  if (addr) {
    return libbirch::LazySharedPtr<bi::type::Object>(context_, reinterpret_cast<make_t*>(addr)(context_));
  } else {
    return libbirch::nil;
  }
  }}
}

/**
 * Make an object.
 *
 *   - Type: A value or class type.
 *
 * Return: if `Type` is a class with no initialization parameters, or a value
 * type, constructs an object of that class or a default-initialized value
 * of that type and returns it in an optional, otherwise returns an optional
 * with no value.
 */
function make<Type>() -> Type? {
  result:Object?; // dummy to ensure context_ is passed to function
  cpp{{
  return libbirch::make<Type>(context_);
  }}
}
