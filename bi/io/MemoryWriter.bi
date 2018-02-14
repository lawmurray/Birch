/**
 * In-memory writer.
 */
class MemoryWriter < Writer {
  hpp{{
  libubjpp::value* group = nullptr;
  }}

  function setObject() {
    cpp{{
    group->set(libubjpp::object_type());
    }}
  }

  function setArray() {
    cpp{{
    group->set(libubjpp::array_type());
    }}
  }

  function setBoolean(value:Boolean) {
    cpp{{
    group->set(value_);
    }}
  }
  
  function setInteger(value:Integer) {
    cpp{{
    group->set(value_);
    }}
  }
  
  function setReal(value:Real) {
    cpp{{
    group->set(value_);
    }}
  }

  function setString(value:String) {
    cpp{{
    group->set(value_);
    }}
  }

  function setBooleanArray(value:Boolean[_]) {
    setBooleanArray([], value);
  }
  
  function setIntegerArray(value:Integer[_]) {
    setIntegerArray([], value);
  }
  
  function setRealArray(value:Real[_]) {
    setRealArray([], value);
  }

  function setStringArray(value:String[_]) {
    setStringArray([], value);
  }

  function setObject(name:String) {
    cpp{{
    group->set(name_, libubjpp::object_type());
    }}
  }

  function setArray(name:String) {
    cpp{{
    group->set(name_, libubjpp::array_type());
    }}
  }

  function setBoolean(name:String, value:Boolean) {
    cpp{{
    group->set(name_, value_);
    }}
  }
  
  function setInteger(name:String, value:Integer) {
    cpp{{
    group->set(name_, value_);
    }}
  }
  
  function setReal(name:String, value:Real) {
    cpp{{
    group->set(name_, value_);
    }}
  }

  function setString(name:String, value:String) {
    cpp{{
    group->set(name_, value_);
    }}
  }

  function setBooleanArray(name:String, value:Boolean[_]) {
    setBooleanArray([name], value);
  }
  
  function setIntegerArray(name:String, value:Integer[_]) {
    setIntegerArray([name], value);
  }
  
  function setRealArray(name:String, value:Real[_]) {
    setRealArray([name], value);
  }

  function setStringArray(name:String, value:String[_]) {
    setStringArray([name], value);
  }
  
  function setObject(path:[String]) {
    cpp{{
    group->set(path_, libubjpp::object_type());
    }}
  }

  function setArray(path:[String]) {
    cpp{{
    group->set(path_, libubjpp::array_type());
    }}
  }
  
  function setBoolean(path:[String], value:Boolean) {
    cpp{{
    group->set(path_, value_);
    }}
  }
  
  function setInteger(path:[String], value:Integer) {
    cpp{{
    group->set(path_, value_);
    }}
  }
  
  function setReal(path:[String], value:Real) {
    cpp{{
    group->set(path_, value_);
    }}
  }

  function setString(path:[String], value:String) {
    cpp{{
    group->set(path_, value_);
    }}
  }

  function setBooleanArray(path:[String], value:Boolean[_]) {
    cpp{{
    libubjpp::array_type array(value_.length(0));
    std::copy(value_.begin(), value_.end(), array.begin());
    group->set(path_, std::move(array));
    }}
  }
  
  function setIntegerArray(path:[String], value:Integer[_]) {
    cpp{{
    libubjpp::array_type array(value_.length(0));
    std::copy(value_.begin(), value_.end(), array.begin());
    group->set(path_, std::move(array));
    }}
  }
  
  function setRealArray(path:[String], value:Real[_]) {
    cpp{{
    libubjpp::array_type array(value_.length(0));
    std::copy(value_.begin(), value_.end(), array.begin());
    group->set(path_, std::move(array));
    }}
  }

  function setStringArray(path:[String], value:String[_]) {
    cpp{{
    libubjpp::array_type array(value_.length(0));
    std::copy(value_.begin(), value_.end(), array.begin());
    group->set(path_, std::move(array));
    }}
  }

  function push() -> Writer {
    result:MemoryWriter;
    cpp{{
    auto array = group->get<libubjpp::array_type>();
    assert(array);
    array.get().push_back(libubjpp::nil_type());
    result_->group = &array.get().back();
    ///@todo Writer object will be invalid if array resized again
    }}
    return result;
  }
}
