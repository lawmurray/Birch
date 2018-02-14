/**
 * In-memory reader.
 */
class MemoryReader < Reader {
  hpp{{
  libubjpp::value* group = nullptr;
  }}  

  function getObject() -> Reader? {
    exists:Boolean;
    cpp{{
    auto value = group->get<libubjpp::object_type>();
    exists_ = static_cast<bool>(value);
    }}
    if (exists) {
      return this;
    } else {
      return nil;
    }
  }

  fiber getArray() -> Reader! {
    /* fibers do not preserve variables declared in raw C++ code between
     * yields, and the switch and jump mechanism for resumption cannot jump
     * over variable initialization; so the value variable below is
     * recreated temporarily when needed, and put out of scope immediately
     * after */
    length:Integer;
    result:MemoryReader;
    cpp{{
    {
      auto value = this->self()->group->get<libubjpp::array_type>();
      length_ = value ? value.get().size() : 0;
    }
    }}
    for (i:Integer in 1..length) {
      cpp{{
      {
        auto value = this->self()->group->get<libubjpp::array_type>();
        assert(value);
        result_->group = &value.get()[i_ - 1];
      }
      }}
      yield result;
    }
  }

  function getLength() -> Integer? {
    cpp{{
    auto array = group->get<libubjpp::array_type>();
    if (array) {
      return array.get().size();
    } else {
      return nullptr;
    }
    }}
  }

  function getBoolean() -> Boolean? {
    cpp{{
    return group->get<libubjpp::bool_type>();
    }}
  }
  
  function getInteger() -> Integer? {
    cpp{{
    return group->get<libubjpp::int64_type>();
    }}
  }
  
  function getReal() -> Real? {
    cpp{{
    return group->get<libubjpp::double_type>();
    }}
  }
  
  function getString() -> String? {
    cpp{{
    return group->get<libubjpp::string_type>();
    }}
  }

  function getBooleanArray() -> Boolean[_]? {
    return getBooleanArray([]);
  }
  
  function getIntegerArray() -> Integer[_]? {
    return getIntegerArray([]);
  }
  
  function getRealArray() -> Real[_]? {
    return getRealArray([]);
  }
  
  function getStringArray() -> String[_]? {
    return getStringArray([]);
  }

  function getObject(name:String) -> Reader? {
    exists:Boolean;
    cpp{{
    auto value = group->get(name_);
    exists_ = static_cast<bool>(value);
    }}
    if (exists) {
      result:MemoryReader;
      cpp{{
      result_->group = &value.get();
      }}
      return result;
    } else {
      return nil;
    }
  }

  fiber getArray(name:String) -> Reader! {
    length:Integer;
    result:MemoryReader;
    cpp{{
    {
      auto value = this->self()->group->get<libubjpp::array_type>(name_);
      length_ = value ? value.get().size() : 0;
    }
    }}
    for (i:Integer in 1..length) {
      cpp{{
      {
        auto value = this->self()->group->get<libubjpp::array_type>(name_);
        assert(value);
        result_->group = &value.get()[i_ - 1];
      }
      }}
      yield result;
    }
  }

  function getLength(name:String) -> Integer? {
    cpp{{
    auto array = group->get<libubjpp::array_type>(name_);
    if (array) {
      return array.get().size();
    } else {
      return nullptr;
    }
    }}
  }

  function getBoolean(name:String) -> Boolean? {
    cpp{{
    return group->get<libubjpp::bool_type>(name_);
    }}
  }
  
  function getInteger(name:String) -> Integer? {
    cpp{{
    return group->get<libubjpp::int64_type>(name_);
    }}
  }
  
  function getReal(name:String) -> Real? {
    cpp{{
    return group->get<libubjpp::double_type>(name_);
    }}
  }
  
  function getString(name:String) -> String? {
    cpp{{
    return group->get<libubjpp::string_type>(name_);
    }}
  }

  function getBooleanArray(name:String) -> Boolean[_]? {
    return getBooleanArray([name]);
  }

  function getIntegerArray(name:String) -> Integer[_]? {
    return getIntegerArray([name]);
  }

  function getRealArray(name:String) -> Real[_]? {
    return getRealArray([name]);
  }

  function getStringArray(name:String) -> String[_]? {
    return getStringArray([name]);
  }

  function getObject(path:[String]) -> Reader? {
    exists:Boolean;
    cpp{{
    auto value = group->get(path_);
    exists_ = static_cast<bool>(value);
    }}
    if (exists) {
      result:MemoryReader;
      cpp{{
      result_->group = &value.get();
      }}
      return result;
    } else {
      return nil;
    }
  }

  fiber getArray(path:[String]) -> Reader! {
    length:Integer;
    result:MemoryReader;
    cpp{{
    {
      auto value = this->self()->group->get<libubjpp::array_type>(path_);
      length_ = value ? value.get().size() : 0;
    }
    }}
    for (i:Integer in 1..length) {
      cpp{{
      {
        auto value = this->self()->group->get<libubjpp::array_type>(path_);
        assert(value);
        result_->group = &value.get()[i_ - 1];
      }
      }}
      yield result;
    }
  }

  function getLength(path:[String]) -> Integer? {
    cpp{{
    auto array = group->get<libubjpp::array_type>(path_);
    if (array) {
      return array.get().size();
    } else {
      return nullptr;
    }
    }}
  }

  function getBoolean(path:[String]) -> Boolean? {
    cpp{{
    return group->get<libubjpp::bool_type>(path_);
    }}
  }
  
  function getInteger(path:[String]) -> Integer? {
    cpp{{
    return group->get<libubjpp::int64_type>(path_);
    }}
  }
  
  function getReal(path:[String]) -> Real? {
    cpp{{
    return group->get<libubjpp::double_type>(path_);
    }}
  }

  function getString(path:[String]) -> String? {
    cpp{{
    return group->get<libubjpp::string_type>(path_);
    }}
  }

  function getBooleanArray(path:[String]) -> Boolean[_]? {
    length:Integer? <- getLength(path);
    if (length?) {
      cpp{{
      auto array = group->get<libubjpp::array_type>(path_).get();
      }}
      result:Boolean[length!];
      value:Boolean?;
      for (i:Integer in 1..length!) {
        cpp{{
        value_ = array[i_ - 1].get<libubjpp::bool_type>();
        }}
        if (value?) {
          result[i] <- value!;
        } else {
          return nil;
        }
      }
      return result;
    } else {
      return nil;
    }
  }

  function getIntegerArray(path:[String]) -> Integer[_]? {
    length:Integer? <- getLength(path);
    if (length?) {
      cpp{{
      auto array = group->get<libubjpp::array_type>(path_).get();
      }}
      result:Integer[length!];
      value:Integer?;
      for (i:Integer in 1..length!) {
        cpp{{
        value_ = array[i_ - 1].get<libubjpp::int64_type>();
        }}
        if (value?) {
          result[i] <- value!;
        } else {
          return nil;
        }
      }
      return result;
    } else {
      return nil;
    }
  }

  function getRealArray(path:[String]) -> Real[_]? {
    length:Integer? <- getLength(path);
    if (length?) {
      cpp{{
      auto array = group->get<libubjpp::array_type>(path_).get();
      }}
      result:Real[length!];
      value:Real?;
      for (i:Integer in 1..length!) {
        cpp{{
        value_ = array[i_ - 1].get<libubjpp::double_type>();
        }}
        if (value?) {
          result[i] <- value!;
        } else {
          return nil;
        }
      }
      return result;
    } else {
      return nil;
    }
  }

  function getStringArray(path:[String]) -> String[_]? {
    length:Integer? <- getLength(path);
    if (length?) {
      cpp{{
      auto array = group->get<libubjpp::array_type>(path_).get();
      }}
      result:String[length!];
      value:String?;
      for (i:Integer in 1..length!) {
        cpp{{
        value_ = array[i_ - 1].get<libubjpp::string_type>();
        }}
        if (value?) {
          result[i] <- value!;
        } else {
          return nil;
        }
      }
      return result;
    } else {
      return nil;
    }
  }
}
