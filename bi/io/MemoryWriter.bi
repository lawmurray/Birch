/**
 * In-memory writer.
 */
class MemoryWriter < Writer {
  hpp{{
  libubjpp::value* group = nullptr;
  }}

  function setObject() -> Writer {
    cpp{{
    group->set(libubjpp::object_type());
    }}
    return this;
  }

  function setArray() -> Writer {
    cpp{{
    group->set(libubjpp::array_type());
    }}
    return this;
  }

  function setNil() {
    cpp{{
    group->set(libubjpp::nil_type());
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

  function setBooleanVector(value:Boolean[_]) {
    setBooleanVector([], value);
  }
  
  function setIntegerVector(value:Integer[_]) {
    setIntegerVector([], value);
  }
  
  function setRealVector(value:Real[_]) {
    setRealVector([], value);
  }

  function setBooleanMatrix(value:Boolean[_,_]) {
    setBooleanMatrix([], value);
  }
  
  function setIntegerMatrix(value:Integer[_,_]) {
    setIntegerMatrix([], value);
  }
  
  function setRealMatrix(value:Real[_,_]) {
    setRealMatrix([], value);
  }

  function setObject(name:String) -> Writer {
    result:MemoryWriter;
    cpp{{
    result_->group = &group->set(name_, libubjpp::object_type());
    }}
    return result;
  }

  function setArray(name:String) -> Writer {
    result:MemoryWriter;
    cpp{{
    result_->group = &group->set(name_, libubjpp::array_type());
    }}
    return result;
  }

  function setNil(name:String) {
    cpp{{
    group->set(name_, libubjpp::nil_type());
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

  function setBooleanVector(name:String, value:Boolean[_]) {
    setBooleanVector([name], value);
  }
  
  function setIntegerVector(name:String, value:Integer[_]) {
    setIntegerVector([name], value);
  }
  
  function setRealVector(name:String, value:Real[_]) {
    setRealVector([name], value);
  }

  function setBooleanMatrix(name:String, value:Boolean[_,_]) {
    setBooleanMatrix([name], value);
  }
  
  function setIntegerMatrix(name:String, value:Integer[_,_]) {
    setIntegerMatrix([name], value);
  }
  
  function setRealMatrix(name:String, value:Real[_,_]) {
    setRealMatrix([name], value);
  }

  function setObject(path:[String]) -> Writer {
    result:MemoryWriter;
    cpp{{
    result_->group = &group->set(path_, libubjpp::object_type());
    }}
    return result;
  }

  function setArray(path:[String]) -> Writer {
    result:MemoryWriter;
    cpp{{
    result_->group = &group->set(path_, libubjpp::array_type());
    }}
    return result;
  }
  
  function setNil(path:[String]) {
    cpp{{
    group->set(path_, libubjpp::nil_type());
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

  function setBooleanVector(path:[String], value:Boolean[_]) {
    cpp{{
    libubjpp::array_type array(value_.length(0));
    std::copy(value_.begin(), value_.end(), array.begin());
    group->set(path_, std::move(array));
    }}
  }
  
  function setIntegerVector(path:[String], value:Integer[_]) {
    cpp{{
    libubjpp::array_type array(value_.length(0));
    std::copy(value_.begin(), value_.end(), array.begin());
    group->set(path_, std::move(array));
    }}
  }
  
  function setRealVector(path:[String], value:Real[_]) {
    cpp{{
    libubjpp::array_type array(value_.length(0));
    std::copy(value_.begin(), value_.end(), array.begin());
    group->set(path_, std::move(array));
    }}
  }

  function setBooleanMatrix(path:[String], value:Boolean[_,_]) {
    nrows:Integer <- rows(value);
    ncols:Integer <- columns(value);
    writer:Writer <- setArray(path);
    for (i:Integer in 1..nrows) {
      writer.push().setBooleanVector(value[i,1..ncols]);
    }
  }
  
  function setIntegerMatrix(path:[String], value:Integer[_,_]) {
    nrows:Integer <- rows(value);
    ncols:Integer <- columns(value);
    writer:Writer <- setArray(path);
    for (i:Integer in 1..nrows) {
      writer.push().setIntegerVector(value[i,1..ncols]);
    }
  }
  
  function setRealMatrix(path:[String], value:Real[_,_]) {
    nrows:Integer <- rows(value);
    ncols:Integer <- columns(value);
    writer:Writer <- setArray(path);
    for (i:Integer in 1..nrows) {
      writer.push().setRealVector(value[i,1..ncols]);
    }
  }

  function push() -> Writer {
    result:MemoryWriter;
    cpp{{
    auto array = group->get<libubjpp::array_type>();
    assert(array);
    array.get().push_back(libubjpp::object_type());
    result_->group = &array.get().back();
    }}
    ///@todo Writer object will be invalid if array resized again
    return result;
  }
}
