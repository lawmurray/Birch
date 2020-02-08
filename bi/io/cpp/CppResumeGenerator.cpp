/**
 * @file
 */
#include "bi/io/cpp/CppResumeGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/io/bih_ostream.hpp"
#include "bi/primitive/encode.hpp"

bi::CppResumeGenerator::CppResumeGenerator(const Yield* yield,
    std::ostream& base, const int level, const bool header) :
    CppBaseGenerator(base, level, header), yield(yield) {
  //
}

void bi::CppResumeGenerator::visit(const Function* o) {
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
    auto i = 0;
    for (auto named : yield->state) {
      genSourceLine(o->loc);
      line("auto " << named->name << " = state.get<" << i++ << ">();");
    }
    *this << o->braces->strip();
    out();
    line("}\n");
  }
}

void bi::CppResumeGenerator::visit(const Yield* o) {
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
  middle("resume_" << o->number << "_<decltype(state_" << o->number << "_)>");
  finish(", std::move(state_" << o->number << "_));");
}

void bi::CppResumeGenerator::visit(const Return* o) {
  genTraceLine(o->loc);
  start("return libbirch::make_fiber<yield_type,return_type>");
  finish('(' << o->single << ");");
}

std::string bi::CppResumeGenerator::getName(const std::string& name,
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
