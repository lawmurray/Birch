/**
 * @file
 */
#include "bi/exception/ConversionOperatorException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::ConversionOperatorException::ConversionOperatorException(
    const ConversionOperator* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: conversion operators only support value types\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o;
  msg = base.str();
}
