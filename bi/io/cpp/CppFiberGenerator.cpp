/**
 * @file
 */
#include "bi/io/cpp/CppFiberGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/io/bih_ostream.hpp"
#include "bi/primitive/encode.hpp"

bi::CppFiberGenerator::CppFiberGenerator(std::ostream& base, const int level,
    const bool header) :
    CppBaseGenerator(base, level, header) {

}

void bi::CppFiberGenerator::visit(const Fiber* o) {
  theFiber = o;
  auto fiberType = dynamic_cast<const FiberType*>(theFiber->returnType);
  assert(fiberType);

  /* initial function */
  if (!o->braces->isEmpty()) {
    if (!header) {
      genSourceLine(o->loc);
    }
    genTemplateParams(o);
    if (!header) {
      genSourceLine(o->loc);
    }
    start(o->returnType << ' ');
    if (!header) {
      middle("bi::");
    }
    middle(o->name << '(' << o->params << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      *this << o->yield;
      out();
      line("}\n");
    }

    /* resume functions */
    Gatherer<Yield> yields;
    o->yield->accept(&yields);
    o->accept(&yields);
    for (auto yield : yields) {
      if (yield->resume) {
        *this << yield->resume;
      }
    }
  }
}

void bi::CppFiberGenerator::visit(const Function* o) {
  if (!header) {
    genSourceLine(o->loc);
  }
  start("template<");
  if (!o->typeParams->isEmpty()) {
    middle(o->typeParams << ", ");
  }
  finish("class State>");
  if (!header) {
    genSourceLine(o->loc);
  }
  start(o->returnType << ' ');
  if (!header) {
    middle("bi::");
  }
  middle(o->name << '_' << o->number << "_(const State& state) {");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genTraceFunction(o->name->str(), o->loc);
    *this << o->braces->strip();
    out();
    line("}\n");
  }
}

void bi::CppFiberGenerator::visit(const Yield* o) {
  auto fiberType = dynamic_cast<const FiberType*>(theFiber->returnType);
  assert(fiberType);
  genTraceLine(o->loc);
  start("auto state_" << o->number << "_ = libbirch::make_tuple(");
  for (auto iter = o->state.begin(); iter != o->state.end(); ++iter) {
    if (iter != o->state.begin()) {
      middle(", ");
    }
    middle(*iter);
  }
  finish(");");

  start("return libbirch::make_fiber");
  middle('<' << fiberType->yieldType << ',' << fiberType->returnType << '>');
  middle('(');
  if (!o->single->isEmpty()) {
    middle(o->single << ", ");
  }
  middle(theFiber->name << '_' << o->number << "_<");
  if (!theFiber->typeParams->isEmpty()) {
    middle(theFiber->typeParams << ", ");
  }
  middle("decltype(state_" << o->number << "_)>");
  finish(", std::move(state_" << o->number << "_));");
}

void bi::CppFiberGenerator::visit(const Return* o) {
  auto fiberType = dynamic_cast<const FiberType*>(theFiber->returnType);
  assert(fiberType);
  genTraceLine(o->loc);
  start("return libbirch::make_fiber");
  middle('<' << fiberType->yieldType << ',' << fiberType->returnType << '>');
  finish('(' << o->single << ");");
}

std::string bi::CppFiberGenerator::getName(const std::string& name,
    const int number) {
  std::stringstream buf;
  std::string result;
  auto iter = names.find(number);
  if (iter == names.end()) {
    auto count = counts.find(name);
    if (count == counts.end()) {
      buf << internalise(name);
      result = buf.str();
      counts.insert(std::make_pair(name, 1));
    } else {
      buf << internalise(name) << count->second << '_';
      result = buf.str();
      ++count->second;
    }
    names.insert(std::make_pair(number, result));
  } else {
    result = iter->second;
  }
  return result;
}
