/**
 * Reader for JSON (JavaScript Object Notation) files.
 */
class JSONReader < Reader {
  hpp{{
  /**
   * Top-level entry in the file.
   */
  libubjpp::value top;
  }}

  /**
   * Open file.
   *
   *   - path: Path.
   */
  function open(path:String) {
    cpp{{
    libubjpp::JSONDriver driver;
    auto result = driver.parse(path_);
    if (result) {
      top = result.get();
    }
    }}
  }

  function' readBoolean(name:String) -> Boolean? {
    result:Boolean?;
    cpp{{
    auto value = top.get<libubjpp::bool_type>(name_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function' readInteger(name:String) -> Integer? {
    result:Integer?;
    cpp{{
    auto value = top.get<libubjpp::int64_type>(name_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function' readReal(name:String) -> Real? {
    result:Real?;
    cpp{{
    auto value = top.get<libubjpp::double_type>(name_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function' readBoolean(path:[String]) -> Boolean? {
    result:Boolean?;
    cpp{{
    auto value = top.get<libubjpp::bool_type>(path_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function' readInteger(path:[String]) -> Integer? {
    result:Integer?;
    cpp{{
    auto value = top.get<libubjpp::int64_type>(path_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function' readReal(path:[String]) -> Real? {
    result:Real?;
    cpp{{
    auto value = top.get<libubjpp::double_type>(path_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
}
