/**
 * @file
 */
#include "bi/io/cpp/CppFiberGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/encode.hpp"

bi::CppFiberGenerator::CppFiberGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header),
    fiber(nullptr),
    yield(nullptr) {
  //
}

void bi::CppFiberGenerator::visit(const Fiber* o) {
  if (!o->braces->isEmpty()) {
    this->fiber = o;

    /* initial function */
    genTemplateParams(o);
    genSourceLine(o->loc);
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
        this->yield = yield;
        *this << this->yield->resume;
      }
    }
  }
}

void bi::CppFiberGenerator::visit(const Function* o) {
  auto fiberType = dynamic_cast<const FiberType*>(fiber->returnType);
  assert(fiberType);
  genSourceLine(o->loc);
  start("template<");
  for (auto typeParam : *o->typeParams) {
    middle("class " << typeParam << ',');
  }
  finish("class State>");
  genSourceLine(o->loc);
  start(o->returnType << ' ');
  if (!header) {
    middle("bi::");
  }
  middle(o->name << '_' << o->number << "_(const State& state)");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genTraceFunction(o->name->str(), o->loc);
    line("libbirch_global_start_");
    genSourceLine(o->loc);
    line("using yield_type_ = " << fiberType->yieldType << ';');
    genSourceLine(o->loc);
    line("using return_type_ = " << fiberType->returnType << ';');
    if (this->yield) {
      auto i = 0;
      for (auto named : this->yield->state) {
        genSourceLine(o->loc);
        start("auto " << getName(named->name->str(), named->number));
        finish(" = state.get<" << i++ << ">();");
      }
    }
    *this << o->braces->strip();
    out();
    line("}\n");
  }
}

void bi::CppFiberGenerator::visit(const Yield* o) {
  genTraceLine(o->loc);
  start("auto state_" << o->number << "_ = libbirch::make_tuple(");
  for (auto iter = o->state.begin(); iter != o->state.end(); ++iter) {
    if (iter != o->state.begin()) {
      middle(", ");
    }
    middle(*iter);
  }
  finish(");");
  start("return libbirch::make_fiber<yield_type,return_type>(");
  if (!o->single->isEmpty()) {
    middle(o->single << ", ");
  }
  middle(fiber->name << '_' << o->number << "_<");
  if (!fiber->typeParams->isEmpty()) {
    middle(fiber->typeParams << ", ");
  }
  middle("decltype(state_" << o->number << "_)>, ");
  finish("std::move(state_" << o->number << "_));");
}

void bi::CppFiberGenerator::visit(const Return* o) {
  genTraceLine(o->loc);
  start("return libbirch::make_fiber<yield_type,return_type>");
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
