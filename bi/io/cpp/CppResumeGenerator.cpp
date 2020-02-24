/**
 * @file
 */
#include "bi/io/cpp/CppResumeGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/encode.hpp"

bi::CppResumeGenerator::CppResumeGenerator(const Class* currentClass,
    const Fiber* currentFiber, std::ostream& base, const int level,
    const bool header) :
    CppBaseGenerator(base, level, header),
    currentClass(currentClass),
    currentFiber(currentFiber),
    stateIndex(0) {
  //
}

void bi::CppResumeGenerator::visit(const Function* o) {
  auto fiberType = dynamic_cast<const FiberType*>(o->returnType);
  assert(fiberType);
  auto generic = o->typeParams->width() + o->params->width() > 0;
  auto args = requiresState(o);

  genSourceLine(o->loc);
  if (args || generic) {
    start("template<");
  }
  for (auto typeParam : *o->typeParams) {
    middle("class " << typeParam << ',');
  }
  if (args) {
    middle("class State_");
  }
  if (args || generic) {
    finish('>');
  }
  if (header) {
    genSourceLine(o->loc);
    start("struct ");
    genUniqueName(o);
    middle(" final : public libbirch::FiberState<" << fiberType->yieldType << ',');
    finish(fiberType->returnType << "> {");
    in();

    if (args) {
      genSourceLine(o->loc);
      line("State_ state_;\n");

      genSourceLine(o->loc);
      start("");
      genUniqueName(o);
      finish("(const State_& state_) : state_(state_) {");
      in();
      line("//");
      out();
      line("}\n");
    }

    genSourceLine(o->loc);
    line("virtual " << fiberType << " query();");
    start("LIBBIRCH_CLASS(");
    genUniqueName(o);
    middle(", libbirch::FiberState<" << fiberType->yieldType << ',');
    finish(fiberType->returnType << ">)");
    if (!o->params->isEmpty()) {
      line("LIBBIRCH_MEMBERS(state_)");
    } else {
      line("LIBBIRCH_MEMBERS()");
    }
    out();
    line("};\n");
    genYieldMacro(o);
  } else {
    genSourceLine(o->loc);
    start(fiberType << " bi::");
    genUniqueName(o);
    if (args || generic) {
      start('<');
    }
    for (auto typeParam : *o->typeParams) {
      middle("class " << typeParam << ',');
    }
    if (args) {
      middle("State_");
    }
    if (args || generic) {
      middle('>');
    }
    finish("::query() {");
    in();
    genTraceFunction(o->name->str(), o->loc);
    genUnpack(o);
    *this << o->braces->strip();
    out();
    line("}");
  }
}

void bi::CppResumeGenerator::visit(const MemberFunction* o) {
  auto fiberType = dynamic_cast<const FiberType*>(o->returnType);
  assert(fiberType);

  if (!header) {
    genTemplateParams(currentClass);
  }
  genSourceLine(o->loc);
  start("template<");
  for (auto typeParam : *o->typeParams) {
    middle("class " << typeParam << ',');
  }
  finish("class State_>");
  if (header) {
    genSourceLine(o->loc);
    start("struct ");
    genUniqueName(o);
    middle(" final : public libbirch::FiberState<" << fiberType->yieldType << ',');
    finish(fiberType->returnType << "> {");
    in();

    genSourceLine(o->loc);
    line("State_ state_;\n");

    genSourceLine(o->loc);
    start("");
    genUniqueName(o);
    finish("(const State_& state_) : state_(state_) {");
    in();
    line("//");
    out();
    line("}\n");

    genSourceLine(o->loc);
    line("virtual " << fiberType << " query();");
    start("LIBBIRCH_CLASS(");
    genUniqueName(o);
    middle(", libbirch::FiberState<" << fiberType->yieldType << ',');
    finish(fiberType->returnType << ">)");
    if (!o->params->isEmpty()) {
      line("LIBBIRCH_MEMBERS(state_)");
    } else {
      line("LIBBIRCH_MEMBERS()");
    }
    out();
    line("};\n");
    genYieldMacro(o);
  } else {
    genSourceLine(o->loc);
    start(fiberType << " bi::type::" << currentClass->name);
    genTemplateArgs(currentClass);
    middle("::");
    genUniqueName(o);
    start('<');
    for (auto typeParam : *o->typeParams) {
      middle("class " << typeParam << ',');
    }
    finish("State_>::query() {");
    in();
    genTraceFunction(o->name->str(), o->loc);
    genUnpack(o);
    *this << o->braces->strip();
    out();
    line("}");
  }
}

void bi::CppResumeGenerator::visit(const Yield* o) {
  genTraceLine(o->loc);
  start("yield_");
  genUniqueName(o);
  finish('(' << o->single << ");");
}

void bi::CppResumeGenerator::visit(const Return* o) {
  genTraceLine(o->loc);
  start("return " << currentFiber->returnType << '(');
  if (!o->single->isEmpty()) {
    middle(o->single);
  }
  finish(");");
}

void bi::CppResumeGenerator::visit(const LocalVariable* o) {
  if (o->has(RESUME)) {
    start("auto " << getName(o->name->str(), o->number));
    finish(" = state_.template get<" << stateIndex++ << ">();");
  } else {
    CppBaseGenerator::visit(o);
  }
}

void bi::CppResumeGenerator::genUniqueName(const Numbered* o) {
  middle(currentFiber->name << '_' << currentFiber->number << '_');
  middle(o->number << '_');
}

void bi::CppResumeGenerator::genPackType(const Function* o) {
  bool hasPackType = !currentFiber->typeParams->isEmpty() || requiresState(o);
  if (hasPackType) {
    middle('<');
  }
  if (!currentFiber->typeParams->isEmpty()) {
    middle(currentFiber->typeParams);
  }
  if (requiresState(o)) {
    middle("decltype(");
    genPack(o);
    middle(')');
  }
  if (hasPackType) {
    middle('>');
  }
}

void bi::CppResumeGenerator::genPack(const Function* o) {
  Gatherer<Parameter> params;
  o->accept(&params);

  Gatherer<LocalVariable> locals([](auto o) { return o->has(RESUME); });
  o->accept(&locals);

  bool requiresPack = currentClass || (params.size() + locals.size() > 0);
  bool first = true;

  if (requiresPack) {
    middle("libbirch::make_tuple(");
  }
  if (currentClass) {
    middle("self");
    first = false;
  }
  for (auto param : params) {
    if (!first) {
      middle(", ");
    }
    middle(getName(param->name->str(), param->number));
    first = false;
  }
  for (auto local : locals) {
    if (!first) {
      middle(", ");
    }
    middle(getName(local->name->str(), local->number));
    first = false;
  }
  if (requiresPack) {
    middle(')');
  }
}

void bi::CppResumeGenerator::genUnpack(const Function* o) {
  Gatherer<Parameter> params;
  o->params->accept(&params);

  if (currentClass) {
    line("auto self = state_.template get<" << stateIndex++ << ">();");
  }
  for (auto param : params) {
    genSourceLine(param->loc);
    start("auto " << getName(param->name->str(), param->number));
    finish(" = state_.template get<" << stateIndex++ << ">();");
  }
}

void bi::CppResumeGenerator::genYieldMacro(const Function* o) {
  start("#define yield_");
  genUniqueName(o);
  middle('(');
  if (!o->has(START)) {
    middle("...");
  }
  middle(") return " << currentFiber->returnType << '(');
  if (!o->has(START)) {
    middle("__VA_ARGS__");
  }
  if (!o->has(START)) {
    middle(", ");
  }
  middle("new ");
  //if (currentClass && !currentClass->typeParams->isEmpty()) {
  //  middle("typename " << currentClass->name << '<' << currentClass->typeParams << ">::");
  //}
  genUniqueName(o);
  genPackType(o);
  middle('(');
  genPack(o);
  middle(')');
  finish(")\n");
}

bool bi::CppResumeGenerator::requiresState(const Function* o) {
  Gatherer<Parameter> params;
  o->accept(&params);

  Gatherer<LocalVariable> locals([](auto o) { return o->has(RESUME); });
  o->accept(&locals);

  return currentClass || (params.size() + locals.size() > 0);
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
      buf << internalise(name) << '_' << count->second << '_';
      result = buf.str();
      ++count->second;
    }
    names.insert(std::make_pair(number, result));
  } else {
    result = iter->second;
  }
  return result;
}
