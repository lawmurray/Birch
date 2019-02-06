/*
 * Auxiliary class for MemoryBuffer. This should only be used as a temporary
 * within MemoryBuffer. It contains a C++ member variable pointer to a nested
 * C++ member variable in MemoryBuffer that is not clone safe.
 */
class MemoryBufferAuxiliary < Buffer {
  hpp{{
  libubjpp::value* group;
  }}

  function get() -> Buffer {
    return this;
  }

  function getObject() -> Buffer? {
    exists:Boolean;
    cpp{{
    auto object = group->get<libubjpp::object_type>();
    exists_ = static_cast<bool>(object);
    }}
    if (exists) {
      return this;
    } else {
      return nil;
    }
  }

  function getLength() -> Integer? {
    exists:Boolean <- false;
    length:Integer <- 0;
    cpp{{
    auto array = group->get<libubjpp::array_type>();
    exists_ = static_cast<bool>(array);
    if (exists_) {
      length_ = array.get().size();
    }
    }}
    if (exists) {
      return length;
    } else {
      return nil;
    }
  }

  function getArray() -> Buffer? {
    exists:Boolean;
    cpp{{
    auto array = group->get<libubjpp::array_type>();
    exists_ = static_cast<bool>(array);
    }}
    if (exists) {
      return this;
    } else {
      return nil;
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
    auto value1 = group->get<libubjpp::double_type>();
    if (value1) {
      result_ = value1.get();
    } else {
      auto value2 = group->get<libubjpp::int64_type>();
      if (value2) {
        result_ = value2.get();
      }
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

  function getObject(value:Object) -> Object? {
    value.read(this);
    return value;
  }

  function getBooleanVector() -> Boolean[_]? {
    length:Integer? <- getLength();
    if (length?) {
      cpp{{
      auto array = group->get<libubjpp::array_type>().get();
      }}
      result:Boolean[length!];
      value:Boolean?;
      for (i:Integer in 1..length!) {
        cpp{{
        auto value = array[i_ - 1].get<libubjpp::bool_type>();
        if (value) {
          value_ = value.get();
        }
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

  function getIntegerVector() -> Integer[_]? {
    length:Integer? <- getLength();
    if (length?) {
      cpp{{
      auto array = group->get<libubjpp::array_type>().get();
      }}
      result:Integer[length!];
      value:Integer?;
      for (i:Integer in 1..length!) {
        cpp{{
        auto value = array[i_ - 1].get<libubjpp::int64_type>();
        if (value) {
          value_ = value.get();
        }
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

  function getRealVector() -> Real[_]? {
    length:Integer? <- getLength();
    if (length?) {
      cpp{{
      auto array = group->get<libubjpp::array_type>().get();
      }}
      result:Real[length!];
      for (i:Integer in 1..length!) {
        value1:Real?;
        cpp{{
        auto value1 = array[i_ - 1].get<libubjpp::double_type>();
        if (value1) {
          value1_ = value1.get();
        }
        }}
        if (value1?) {
          result[i] <- value1!;
        } else {
          value2:Integer?;
          cpp{{
          auto value2 = array[i_ - 1].get<libubjpp::int64_type>();
          if (value2) {
            value2_ = value2.get();
          }
          }}
          if (value2?) {
            result[i] <- value2!;
          } else {
            return nil;
          }
        }
      }
      return result;
    } else {
      return nil;
    }
  }

  function getBooleanMatrix() -> Boolean[_,_]? {
    nrows:Integer? <- getLength();
    if (nrows?) {
      row:Buffer! <- walk();
      if (row?) {
        /* determine number of columns from first row */
        ncols:Integer? <- row!.getLength();
        X:Boolean[_,_];
        x:Boolean[_]?;
        if (ncols?) {
          X <- matrix(false, nrows!, ncols!);
          x <- row!.getBooleanVector();
          if (x?) {
            X[1,1..ncols!] <- x!;
          } else {
            return nil;
          }
        }
        
        /* read in remaining rows, requiring that they have the same
         * number of columns as the first */
        i:Integer <- 1;
        while (row?) {
          ncols <- row!.getLength();
          if (ncols? && ncols! == columns(X)) {
            x <- row!.getBooleanVector();
            if (x?) {
              i <- i + 1;
              X[i,1..ncols!] <- x!;
            } else {
              return nil;
            }
          } else {
            return nil;
          }
        }
        assert i == nrows!;
        return X;
      }
    }
    return nil;
  }

  function getIntegerMatrix() -> Integer[_,_]? {
    nrows:Integer? <- getLength();
    if (nrows?) {
      row:Buffer! <- walk();
      if (row?) {
        /* determine number of columns from first row */
        ncols:Integer? <- row!.getLength();
        X:Integer[_,_];
        x:Integer[_]?;
        if (ncols?) {
          X <- matrix(0, nrows!, ncols!);
          x <- row!.getIntegerVector();
          if (x?) {
            X[1,1..ncols!] <- x!;
          } else {
            return nil;
          }
        }
        
        /* read in remaining rows, requiring that they have the same
         * number of columns as the first */
        i:Integer <- 1;
        while (row?) {
          ncols <- row!.getLength();
          if (ncols? && ncols! == columns(X)) {
            x <- row!.getIntegerVector();
            if (x?) {
              i <- i + 1;
              X[i,1..ncols!] <- x!;
            } else {
              return nil;
            }
          } else {
            return nil;
          }
        }
        assert i == nrows!;
        return X;
      }
    }
    return nil;
  }

  function getRealMatrix() -> Real[_,_]? {
    nrows:Integer? <- getLength();
    if (nrows?) {
      row:Buffer! <- walk();
      if (row?) {
        /* determine number of columns from first row */
        ncols:Integer? <- row!.getLength();
        X:Real[_,_];
        x:Real[_]?;
        if (ncols?) {
          X <- matrix(0.0, nrows!, ncols!);
          x <- row!.getRealVector();
          if (x?) {
            X[1,1..ncols!] <- x!;
          } else {
            return nil;
          }
        }
        
        /* read in remaining rows, requiring that they have the same
         * number of columns as the first */
        i:Integer <- 1;
        while (row?) {
          ncols <- row!.getLength();
          if (ncols? && ncols! == columns(X)) {
            x <- row!.getRealVector();
            if (x?) {
              i <- i + 1;
              X[i,1..ncols!] <- x!;
            } else {
              return nil;
            }
          } else {
            return nil;
          }
        }
        assert i == nrows!;
        return X;
      }
    }
    return nil;
  }

  function getChild(name:String) -> Buffer? {
    exists:Boolean <- false;
    cpp{{
    auto child = group->get(name_);
    exists_ = static_cast<bool>(child);
    }}
    if (exists) {
      buffer:MemoryBufferAuxiliary;
      cpp{{
      buffer_->group = &child.get();
      }}
      return buffer;
    } else {
      return nil;
    }
  }

  function set() -> Buffer {
    cpp{{
    group->set();
    }}
    return this;
  }

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
  
  function setNil() {
    cpp{{
    group->set(libubjpp::nil_type());
    }}
  }
  
  function setBoolean(value:Boolean?) {
    if value? {
      cpp{{
      group->set(value_.get());
      }}
    } else {
      setNil();
    }
  }
  
  function setInteger(value:Integer?) {
    if value? {
      cpp{{
      group->set(value_.get());
      }}
    } else {
      setNil();
    }
  }
  
  function setReal(value:Real?) {
    if value? {
      cpp{{
      group->set(value_.get());
      }}
    } else {
      setNil();
    }
  }

  function setString(value:String?) {
    if value? {
      cpp{{
      group->set(value_.get());
      }}
    } else {
      setNil();
    }
  }

  function setObject(value:Object?) {
    if value? {
      cpp{{
      group->set(libubjpp::object_type());
      }}
      value!.write(this);
    } else {
      setNil();
    }
  }

  function setBooleanVector(value:Boolean[_]?) {
    if value? {
      cpp{{
      auto value = value_.get();
      libubjpp::array_type array(value.length(0));
      auto first = value.begin();
      auto last = first + value.size();
      std::copy(first, last, array.begin());
      group->set(std::move(array));
      }}
    } else {
      setNil();
    }
  }
  
  function setIntegerVector(value:Integer[_]?) {
    if value? {
      cpp{{
      auto value = value_.get();
      libubjpp::array_type array(value.length(0));
      auto first = value.begin();
      auto last = first + value.size();
      std::copy(first, last, array.begin());
      group->set(std::move(array));
      }}
    } else {
      setNil();
    }
  }
  
  function setRealVector(value:Real[_]?) {
    if value? {
      cpp{{
      auto value = value_.get();
      libubjpp::array_type array(value.length(0));
      auto first = value.begin();
      auto last = first + value.size();
      std::copy(first, last, array.begin());
      group->set(std::move(array));
      }}
    } else {
      setNil();
    }
  }

  function setBooleanMatrix(value:Boolean[_,_]?) {
    if value? {
      setArray();
      auto nrows <- rows(value!);
      auto ncols <- columns(value!);
      for (i:Integer in 1..nrows) {
        push().setBooleanVector(value![i,1..ncols]);
      }
    } else {
      setNil();
    }
  }
  
  function setIntegerMatrix(value:Integer[_,_]?) {
    if value? {
      setArray();
      auto nrows <- rows(value!);
      auto ncols <- columns(value!);
      for (i:Integer in 1..nrows) {
        push().setIntegerVector(value![i,1..ncols]);
      }
    } else {
      setNil();
    }
  }
  
  function setRealMatrix(value:Real[_,_]?) {
    if value? {
      setArray();
      auto nrows <- rows(value!);
      auto ncols <- columns(value!);
      for (i:Integer in 1..nrows) {
        push().setRealVector(value![i,1..ncols]);
      }
    } else {
      setNil();
    }
  }

  function setChild(name:String) -> Buffer {
    buffer:MemoryBufferAuxiliary;
    cpp{{
    buffer_->group = &group->set(name_);
    }}
    return buffer;
  }

  fiber walk() -> Buffer {
    /* in a fiber here, so have to be careful with the nested C++: variables
     * declared in raw C++ are not preserved between yields, and must use
     * `self` when referring to member variables */
    length:Integer? <- getLength();
    if (length?) {
      for (i:Integer in 1..length!) {
        buffer:MemoryBufferAuxiliary;
        cpp{{
        {
          /* enclosed in local block so that these go out of scope before
           * yield, necessary for fiber implementation */
          auto array = self->group->get<libubjpp::array_type>();
          assert(array);
          buffer_->group = &array.get()[i_ - 1];
        }
        }}
        yield buffer;
      }
    }
  }

  function push() -> Buffer {
    buffer:MemoryBufferAuxiliary;
    cpp{{
    auto array = group->get<libubjpp::array_type>();
    assert(array);
    array.get().push_back(libubjpp::nil_type());
    buffer_->group = &array.get().back();
    }}
    ///@todo Buffer object will be invalid if array resized again
    return buffer;
  }
}
