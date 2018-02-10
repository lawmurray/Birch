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

  function getBoolean() -> Boolean? {
    result:Boolean?;
    cpp{{
    auto value = group->get<libubjpp::bool_type>();
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function getInteger() -> Integer? {
    result:Integer?;
    cpp{{
    auto value = group->get<libubjpp::int64_type>();
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function getReal() -> Real? {
    result:Real?;
    cpp{{
    auto value = group->get<libubjpp::double_type>();
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function getString() -> String? {
    result:String?;
    cpp{{
    auto value = group->get<libubjpp::string_type>();
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
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

  function getBoolean(name:String) -> Boolean? {
    result:Boolean?;
    cpp{{
    auto value = group->get<libubjpp::bool_type>(name_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function getInteger(name:String) -> Integer? {
    result:Integer?;
    cpp{{
    auto value = group->get<libubjpp::int64_type>(name_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function getReal(name:String) -> Real? {
    result:Real?;
    cpp{{
    auto value = group->get<libubjpp::double_type>(name_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function getString(name:String) -> String? {
    result:String?;
    cpp{{
    auto value = group->get<libubjpp::string_type>(name_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
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

  function getBoolean(path:[String]) -> Boolean? {
    result:Boolean?;
    cpp{{
    auto value = group->get<libubjpp::bool_type>(path_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function getInteger(path:[String]) -> Integer? {
    result:Integer?;
    cpp{{
    auto value = group->get<libubjpp::int64_type>(path_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
  
  function getReal(path:[String]) -> Real? {
    result:Real?;
    cpp{{
    auto value = group->get<libubjpp::double_type>(path_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }

  function getString(path:[String]) -> String? {
    result:String?;
    cpp{{
    auto value = group->get<libubjpp::string_type>(path_);
    if (value) {
      result_ = value.get();
    }
    }}
    return result;
  }
}
