/**
 * Buffer in memory.
 */
class MemoryBuffer < Buffer {
  /**
   * The value of this buffer.
   */
  value:Value <- ObjectValue();

  /**
   * Load from a file.
   *
   * - path: Path of the file.
   */
  function load(path:String) {
    auto reader <- Reader(path);
    reader.read(this);
    reader.close();
  }

  /**
   * Save to a JSON file.
   *
   * - path: Path of the file.
   */
  function save(path:String) {
    auto writer <- Writer(path);
    writer.write(this);
    writer.close();
  }

  function getChild(name:String) -> Buffer? {
    return value.getChild(name);
  }

  function setChild(name:String) -> Buffer {
    return value.setChild(name);
  }

  function size() -> Integer {
    return value.size();
  }

  fiber walk() -> Buffer {
    value.walk();
  }

  function push() -> Buffer {
    return value.push();
  }

  function getObject() -> Buffer? {
    if value.isObject() {
      return this;
    } else {
      return nil;
    }
  }

  function getArray() -> Buffer? {
    if value.isArray() {
      return this;
    } else {
      return nil;
    }
  }

  function getBoolean() -> Boolean? {
    return value.getBoolean();
  }
  
  function getInteger() -> Integer? {
    return value.getInteger();
  }
  
  function getReal() -> Real? {
    return value.getReal();
  }

  function getString() -> String? {
    return value.getString();
  }

  function getBooleanVector() -> Boolean[_]? {
    return value.getBooleanVector();
  }

  function getIntegerVector() -> Integer[_]? {
    return value.getIntegerVector();
  }

  function getRealVector() -> Real[_]? {
    return value.getRealVector();
  }

  function getBooleanMatrix() -> Boolean[_,_]? {
    return value.getBooleanMatrix();
  }

  function getIntegerMatrix() -> Integer[_,_]? {
    return value.getIntegerMatrix();
  }

  function getRealMatrix() -> Real[_,_]? {
    return value.getRealMatrix();
  }

  function setObject() -> Buffer {
    v:ObjectValue;
    this.value <- v;
    return this;
  }

  function setArray() -> Buffer {
    v:ArrayValue;
    this.value <- v;
    return this;
  }
  
  function setNil() {
    v:NilValue;
    this.value <- v;
  }
  
  function setBoolean(value:Boolean?) {
    if value? {
      v:BooleanValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }
  
  function setInteger(value:Integer?) {
    if value? {
      v:IntegerValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }
  
  function setReal(value:Real?) {
    if value? {
      v:RealValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }

  function setString(value:String?) {
    if value? {
      v:StringValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }

  function setBooleanVector(value:Boolean[_]?) {
    if value? {
      v:BooleanVectorValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }
  
  function setIntegerVector(value:Integer[_]?) {
    if value? {
      v:IntegerVectorValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }
  
  function setRealVector(value:Real[_]?) {
    if value? {
      v:RealVectorValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }
  
  function setObjectVector(value:Object[_]?) {
    if value? {
      auto v <- value!;
      setArray();
      for n in 1..length(v) {
        push().set(v[n]);
      }
    } else {
      setNil();
    }
  }

  function setBooleanMatrix(value:Boolean[_,_]?) {
    if value? {
      v:BooleanMatrixValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }
  
  function setIntegerMatrix(value:Integer[_,_]?) {
    if value? {
      v:IntegerMatrixValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }
  
  function setRealMatrix(value:Real[_,_]?) {
    if value? {
      v:RealMatrixValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }

  function setObjectMatrix(value:Object[_,_]?) {
    if value? {
      auto v <- value!;
      setArray();
      for i in 1..rows(v) {
        auto buffer <- push().setArray();
        for j in 1..columns(v) {
          buffer.push().set(v[i,j]);
        }
      }
    } else {
      setNil();
    }
  }
}

function MemoryBuffer(value:Value) -> MemoryBuffer {
  o:MemoryBuffer;
  o.value <- value;
  return o;
}
