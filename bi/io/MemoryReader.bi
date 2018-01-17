/**
 * In-memory reader.
 */
class MemoryReader {
  hpp{{
  libubjpp::value* group;
  }}

  function' getBoolean(name:String) -> Boolean? {
    result:Boolean?;
    cpp{{
    auto value = group->get<libubjpp::bool_type>(name_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function' getInteger(name:String) -> Integer? {
    result:Integer?;
    cpp{{
    auto value = group->get<libubjpp::int64_type>(name_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function' getReal(name:String) -> Real? {
    result:Real?;
    cpp{{
    auto value = group->get<libubjpp::double_type>(name_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function' getBoolean(path:[String]) -> Boolean? {
    result:Boolean?;
    cpp{{
    auto value = group->get<libubjpp::bool_type>(path_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function' getInteger(path:[String]) -> Integer? {
    result:Integer?;
    cpp{{
    auto value = group->get<libubjpp::int64_type>(path_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function' getReal(path:[String]) -> Real? {
    result:Real?;
    cpp{{
    auto value = group->get<libubjpp::double_type>(path_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
}
