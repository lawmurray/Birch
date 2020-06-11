/**
 * Buffer in memory.
 */
class MemoryBuffer < Buffer {
  /**
   * The value of this buffer.
   */
  value:Value <- ObjectValue();

  function getChild(name:String) -> Buffer? {
    return value.getChild(name);
  }

  function setChild(name:String) -> Buffer {
    return value.setChild(name);
  }

  function size() -> Integer {
    return value.size();
  }
  
  /**
   * Clear all contents.
   */
  function clear() {
    value <- ObjectValue();
  }

  fiber walk() -> Buffer {
    @value.walk();
  }

  function push() -> Buffer {
    return value.push();
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
  
  function setBoolean(value:Boolean) {
    v:BooleanValue(value);
    this.value <- v;
  }
  
  function setInteger(value:Integer) {
    v:IntegerValue(value);
    this.value <- v;
  }
  
  function setReal(value:Real) {
    v:RealValue(value);
    this.value <- v;
  }

  function setString(value:String) {
    v:StringValue(value);
    this.value <- v;
  }

  function setBooleanVector(value:Boolean[_]) {
    v:BooleanVectorValue(value);
    this.value <- v;
  }
  
  function setIntegerVector(value:Integer[_]) {
    v:IntegerVectorValue(value);
    this.value <- v;
  }
  
  function setRealVector(value:Real[_]) {
    v:RealVectorValue(value);
    this.value <- v;
  }
  
  function setObjectVector(value:Object[_]) {
    setArray();
    for n in 1..length(value) {
      push().set(value[n]);
    }
  }

  function setBooleanMatrix(value:Boolean[_,_]) {
    v:BooleanMatrixValue(value);
    this.value <- v;
  }
  
  function setIntegerMatrix(value:Integer[_,_]) {
    v:IntegerMatrixValue(value);
    this.value <- v;
  }
  
  function setRealMatrix(value:Real[_,_]) {
    v:RealMatrixValue(value);
    this.value <- v;
  }

  function setObjectMatrix(value:Object[_,_]) {
    setArray();
    for i in 1..rows(value) {
      auto buffer <- push().setArray();
      for j in 1..columns(value) {
        buffer.push().set(value[i,j]);
      }
    }
  }
}

function MemoryBuffer(value:Value) -> MemoryBuffer {
  o:MemoryBuffer;
  o.value <- value;
  return o;
}
