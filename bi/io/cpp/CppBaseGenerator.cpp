/**
 * @file
 */
#include "bi/io/cpp/CppBaseGenerator.hpp"

#include "bi/io/cpp/CppClassGenerator.hpp"
#include "bi/io/cpp/CppFiberGenerator.hpp"
#include "bi/io/bih_ostream.hpp"
#include "bi/primitive/encode.hpp"

#include "boost/algorithm/string.hpp"

bi::CppBaseGenerator::CppBaseGenerator(std::ostream& base, const int level,
    const bool header) :
    indentable_ostream(base, level),
    header(header),
    inAssign(0),
    inPointer(0),
    inConstructor(0),
    inLambda(0) {
  //
}

void bi::CppBaseGenerator::visit(const Name* o) {
  middle(internalise(o->str()));
}

void bi::CppBaseGenerator::visit(const ExpressionList* o) {
  middle(o->head);
  if (o->tail) {
    middle(", " << o->tail);
  }
}

void bi::CppBaseGenerator::visit(const Literal<bool>* o) {
  middle(o->str);
}

void bi::CppBaseGenerator::visit(const Literal<int64_t>* o) {
  middle("std::int64_t(" << o->str << ')');
}

void bi::CppBaseGenerator::visit(const Literal<double>* o) {
  middle(o->str);
}

void bi::CppBaseGenerator::visit(const Literal<const char*>* o) {
  middle("std::string(" << o->str << ')');
}

void bi::CppBaseGenerator::visit(const Parentheses* o) {
  if (o->single->type->isList()) {
    if (inAssign) {
      middle("std::tie(" << o->single << ')');
    } else {
      middle("std::make_tuple(" << o->single << ')');
    }
  } else {
    middle('(' << o->single << ')');
  }
}

void bi::CppBaseGenerator::visit(const Sequence* o) {
  if (o->single->isEmpty()) {
    middle("libbirch::nil");
  } else {
    middle("{ " << o->single << " }");
  }
}

void bi::CppBaseGenerator::visit(const Cast* o) {
  if (o->returnType->isClass()) {
    ++inPointer;
    middle("libbirch::dynamic_pointer_cast<" << o->returnType << '>');
  } else {
    middle("libbirch::check_cast<" << o->returnType << '>');
  }
  middle('(' << o->single << ')');
}

void bi::CppBaseGenerator::visit(const Call* o) {
  middle(o->single);
  genArgs(o);
}

void bi::CppBaseGenerator::visit(const BinaryCall* o) {
  auto op = dynamic_cast<OverloadedIdentifier<BinaryOperator>*>(o->single);
  assert(op);
  if (isTranslatable(op->name->str())) {
    /* can use corresponding C++ operator */
    genLeftArg(o);
    middle(' ' << op->name->str() << ' ');
    genRightArg(o);
  } else {
    /* must use as function */
    middle(o->single << '(');
    genLeftArg(o);
    middle(", ");
    genRightArg(o);
    middle(')');
  }
}

void bi::CppBaseGenerator::visit(const UnaryCall* o) {
  auto op = dynamic_cast<OverloadedIdentifier<UnaryOperator>*>(o->single);
  assert(op);
  if (isTranslatable(op->name->str())) {
    /* can use corresponding C++ operator */
    middle(op->name->str());
    genSingleArg(o);
  } else {
    /* must use as function */
    middle(o->single << '(');
    genSingleArg(o);
    middle(')');
  }
}

void bi::CppBaseGenerator::visit(const Assign* o) {
  /* determine whether this is setting a member variable outside of the
   * current class */
  auto member = dynamic_cast<const Member*>(o->left);
  auto slice = dynamic_cast<const Slice*>(o->left);
  const This* self = nullptr;
  const Super* super = nullptr;
  const Identifier<MemberVariable>* var = nullptr;

  if (slice) {
    member = dynamic_cast<const Member*>(slice->single);
  }
  if (member) {
    self = dynamic_cast<decltype(self)>(member->left);
    super = dynamic_cast<decltype(super)>(member->left);
    var = dynamic_cast<decltype(var)>(member->right);
  }

  if (var && !self && !super) {
    /* this assignment is setting a member variable outside of the current
     * class, use the setter member function of the class to ensure the
     * context is set correctly */
    ++inAssign;
    middle(member->left << "->set_" << var->name << "_(");
    if (slice) {
      middle("libbirch::make_view(" << slice->brackets << "), ");
    }
    --inAssign;
    middle(o->right << ')');
  } else {
    ++inAssign;
    middle(o->left);
    --inAssign;
    middle(" = " << o->right);
  }
}

void bi::CppBaseGenerator::visit(const Slice* o) {
  middle(o->single);
  if (!inAssign && o->single->type->isValue()) {
    /* optimization: just reading a value, so ensure that access is in a
     * const context to avoid unnecessary copy */
    middle(".as_const()");
  }
  middle("(libbirch::make_view(" << o->brackets << "))");
}

void bi::CppBaseGenerator::visit(const Query* o) {
  middle(o->single << ".query()");
}

void bi::CppBaseGenerator::visit(const Get* o) {
  middle(o->single);
  if (!o->single->type->isWeak()) {
    // ^ for a weak pointer, the ordinary dereference suffices
    middle(".get()");
  }
}

void bi::CppBaseGenerator::visit(const LambdaFunction* o) {
  middle("[=](libbirch::Label* context_");
  if (!o->params->isEmpty()) {
    middle(", ");
  }
  middle(o->params << ") {");
  in();
  ++inLambda;
  *this << o->braces->strip();
  --inLambda;
  out();
  start("}");
}

void bi::CppBaseGenerator::visit(const Span* o) {
  if (o->single->isEmpty()) {
    middle('0');
  } else {
    middle(o->single);
  }
}

void bi::CppBaseGenerator::visit(const Index* o) {
  middle(o->single << " - 1");
}

void bi::CppBaseGenerator::visit(const Range* o) {
  middle("libbirch::make_range(" << o->left << " - 1, " << o->right << " - 1)");
}

void bi::CppBaseGenerator::visit(const Member* o) {
  auto leftThis = dynamic_cast<const This*>(o->left);
  auto leftSuper = dynamic_cast<const Super*>(o->left);
  if (inConstructor && (leftThis || leftSuper)) {
    if (leftThis) {
      middle("this->");
    } else if (leftSuper) {
      middle("super_type_::");
    }
    middle(o->right);
  } else {
    const Expression* rightVar = dynamic_cast<const Identifier<MemberVariable>*>(o->right);
    if (!rightVar) {
      rightVar = dynamic_cast<const Identifier<LocalVariable>*>(o->right);
      if (!rightVar) {
        rightVar = dynamic_cast<const Identifier<Parameter>*>(o->right);
      }
    }
    middle(o->left);
    if (!inAssign && rightVar && rightVar->type->isValue()) {
      /* optimization: just reading a value, so no need to copy-on-write the
       * owning object */
      middle(".pull()");
    }
    middle("->");

    /* explicitly refer to the super class if necessary */
    auto leftSuper = dynamic_cast<const Super*>(o->left);
    if (leftSuper) {
      middle("super_type_::");
    }
    middle(o->right);
  }
}

void bi::CppBaseGenerator::visit(const This* o) {
  middle("self");
}

void bi::CppBaseGenerator::visit(const Super* o) {
  middle("self");
}

void bi::CppBaseGenerator::visit(const Global* o) {
  middle("::" << o->single);
}

void bi::CppBaseGenerator::visit(const Nil* o) {
  middle("libbirch::nil");
}

void bi::CppBaseGenerator::visit(const Parameter* o) {
  middle("const " << o->type << "& " << o->name);
  if (!o->value->isEmpty()) {
    middle(" = " << o->value);
  }
}

void bi::CppBaseGenerator::visit(const Identifier<Unknown>* o) {
  assert(false);
}

void bi::CppBaseGenerator::visit(const Identifier<Parameter>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const Identifier<GlobalVariable>* o) {
  middle("bi::" << o->name << "()");
}

void bi::CppBaseGenerator::visit(const Identifier<LocalVariable>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const Identifier<MemberVariable>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const OverloadedIdentifier<Function>* o) {
  auto name = internalise(o->name->str());
  if (o->overload->isInstantiation()) {
    std::stringstream base;
    bih_ostream buf(base);
    buf << o->overload->typeParams << '(' << o->overload->params->type << ')';
    name += "_" + encode32(base.str()) + "_";
  }
  middle("bi::" << name);
}

void bi::CppBaseGenerator::visit(const OverloadedIdentifier<Fiber>* o) {
  auto name = internalise(o->name->str());
  if (o->overload->isInstantiation()) {
    std::stringstream base;
    bih_ostream buf(base);
    buf << o->overload->typeParams << '(' << o->overload->params->type << ')';
    name += "_" + encode32(base.str()) + "_";
  }
  middle("bi::" << name);
}

void bi::CppBaseGenerator::visit(
    const OverloadedIdentifier<MemberFunction>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const OverloadedIdentifier<MemberFiber>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(
    const OverloadedIdentifier<BinaryOperator>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(
    const OverloadedIdentifier<UnaryOperator>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const File* o) {
  *this << o->root;
}

void bi::CppBaseGenerator::visit(const GlobalVariable* o) {
  /* C++ does not guarantee static initialization order across compilation
   * units. Global variables are therefore used through accessor functions
   * that initialize their values on first use. */
  start(o->type << "& ");
  if (!header) {
    middle("bi::");
  }
  middle(o->name << "()");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    start("static " << o->type << " result");
    genInit(o);
    finish(';');
    line("return result;");
    out();
    line("}\n");
  }
}

void bi::CppBaseGenerator::visit(const LocalVariable* o) {
  middle(o->type << ' ' << o->name);
  genInit(o);
}

void bi::CppBaseGenerator::visit(const MemberVariable* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const Function* o) {
  if (!o->braces->isEmpty() && o->isBound()) {
    auto name = internalise(o->name->str());
    if (o->isInstantiation()) {
      std::stringstream base;
      bih_ostream buf(base);
      buf << o->typeParams << '(' << o->params->type << ')';
      name += "_" + encode32(base.str()) + "_";
    }
    start(o->returnType << ' ');
    if (!header) {
      middle("bi::");
    }
    middle(name << "(libbirch::Label* context_");
    if (!o->params->isEmpty()) {
      middle(", " << o->params);
    }
    middle(')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceFunction(o->name->str(), o->loc);
      CppBaseGenerator aux(base, level, false);
      *this << o->braces->strip();
      out();
      line("}\n");
    }
  }
  for (auto instantiation : o->instantiations) {
    *this << instantiation;
  }
}

void bi::CppBaseGenerator::visit(const Fiber* o) {
  if (header || o->isBound()) {
    CppFiberGenerator auxFiber(base, level, header);
    auxFiber << o;
  }
  for (auto instantiation : o->instantiations) {
    *this << instantiation;
  }
}

void bi::CppBaseGenerator::visit(const MemberFunction* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const MemberFiber* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const Program* o) {
  if (header) {
    line("extern \"C\" int " << o->name << "(int argc, char** argv);");
  } else {
    line("int bi::" << o->name << "(int argc_, char** argv_) {");
    in();
    genTraceFunction(o->name->str(), o->loc);

    /* initial context */
    line("auto context_ = root_;");

    /* handle program options */
    if (o->params->width() > 0) {
      /* option variables */
      for (auto iter = o->params->begin(); iter != o->params->end(); ++iter) {
        auto param = dynamic_cast<const Parameter*>(*iter);
        assert(param);
        start(param->type << ' ' << param->name);
        if (!param->value->isEmpty()) {
          middle(" = " << param->value);
        } else if (param->type->isClass()) {
          ++inPointer;
          middle(" = libbirch::make_object<" << param->type << ">(context_)");
        }
        finish(';');
      }
      line("");

      /* option flags */
      line("enum {");
      in();
      for (auto param : *o->params) {
        auto name = dynamic_cast<const Parameter*>(param)->name;
        std::string flag = internalise(name->str()) + "FLAG_";
        line(flag << ',');
      }
      out();
      line("};");

      /* long options */
      line("int c_, option_index_;");
      line("option long_options_[] = {");
      in();
      for (auto param : *o->params) {
        auto name = dynamic_cast<const Parameter*>(param)->name;
        std::string flag = internalise(name->str()) + "FLAG_";

        std::string option = name->str();
        boost::replace_all(option, "_", "-");

        start("{\"");
        middle(option << "\", required_argument, 0, " << flag);
        finish(" },");
      }
      line("{0, 0, 0, 0}");
      out();
      line("};");

      /* short options */
      line("const char* short_options_ = \"\";");

      /* read in options with getopt_long */
      line("::opterr = 0;");  // handle error reporting ourselves
      start("c_ = ::getopt_long_only(argc_, argv_, short_options_, ");
      finish("long_options_, &option_index_);");
      line("while (c_ != -1) {");
      in();
      line("switch (c_) {");
      in();

      for (auto param : *o->params) {
        auto name = dynamic_cast<const Parameter*>(param)->name;
        std::string flag = internalise(name->str()) + "FLAG_";

        line("case " << flag << ':');
        in();
        if (param->type->unwrap()->isBasic()) {
          auto type = dynamic_cast<Named*>(param->type->unwrap());
          assert(type);
          start(name << " = bi::" << type->name);
          finish("(std::string(::optarg));");
        } else {
          line(name << " = std::string(::optarg);");
        }
        line("break;");
        out();
      }
      line("default:");
      in();
      line("libbirch::unknown_option(argv_[::optind - 1]);");
      out();
      out();
      line('}');
      start("c_ = ::getopt_long_only(argc_, argv_, short_options_, ");
      finish("long_options_, &option_index_);");
      out();
      line("}\n");
    }

    /* seed random number generator with random entropy */
    line("bi::seed();\n");

    /* body of program */
    if (!o->braces->isEmpty()) {
      CppBaseGenerator aux(base, level, header);
      aux << o->braces->strip();
    }

    line("return 0;");
    out();
    line("}\n");
  }
}

void bi::CppBaseGenerator::visit(const BinaryOperator* o) {
  if (!o->braces->isEmpty()) {
    start(o->returnType << ' ');
    if (!header) {
      middle("bi::");
    }
    if (isTranslatable(o->name->str())) {
      middle("operator" << o->name->str());
    } else {
      middle(o->name);
    }
    middle('(');
    middle(o->params->getLeft() << ", " << o->params->getRight());
    middle(')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceFunction(o->name->str(), o->loc);
      CppBaseGenerator aux(base, level, false);
      aux << o->braces->strip();
      out();
      finish("}\n");
    }
  }
}

void bi::CppBaseGenerator::visit(const UnaryOperator* o) {
  if (!o->braces->isEmpty()) {
    start(o->returnType << ' ');
    if (!header) {
      middle("bi::");
    }
    if (isTranslatable(o->name->str())) {
      middle("operator" << o->name->str());
    } else {
      middle(o->name);
    }
    middle('(' << o->params << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceFunction(o->name->str(), o->loc);
      CppBaseGenerator aux(base, level, false);
      aux << o->braces->strip();
      out();
      finish("}\n");
    }
  }
}

void bi::CppBaseGenerator::visit(const AssignmentOperator* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const ConversionOperator* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const Basic* o) {
  if (header && o->isAlias()) {
    line("using " << o->name << " = " << o->base << ';');
  }
}

void bi::CppBaseGenerator::visit(const Class* o) {
  CppClassGenerator auxClass(base, level, header);
  auxClass << o;
}

void bi::CppBaseGenerator::visit(const Generic* o) {
  if (o->type->isEmpty()) {
    middle(o->name);
  } else {
    middle(o->type);
  }
}

void bi::CppBaseGenerator::visit(const Assume* o) {
  assert(false);  // should have been replaced by Resolver
}

void bi::CppBaseGenerator::visit(const ExpressionStatement* o) {
  genTraceLine(o->loc->firstLine);
  line(o->single << ';');
}

void bi::CppBaseGenerator::visit(const If* o) {
  genTraceLine(o->loc->firstLine);
  start("if (");
  auto cond = o->cond->strip();
  if (cond->type->isClass()) {
    middle("*(");
  }
  middle(cond);
  if (cond->type->isClass()) {
    middle(')');
  }
  finish(") {");
  in();
  *this << o->braces->strip();
  out();
  if (!o->falseBraces->isEmpty()) {
    line("} else {");
    in();
    *this << o->falseBraces->strip();
    out();
  }
  line("}");
}

void bi::CppBaseGenerator::visit(const For* o) {
  genTraceLine(o->loc->firstLine);

  /* handle parallel for loop */
  if (o->has(PARALLEL)) {
    line("#pragma omp parallel");
    line("{");
    in();
    genTraceFunction("<thread start>", o->loc);
    start("#pragma omp for schedule(");
    if (o->has(DYNAMIC)) {
      middle("guided");
    } else {
      middle("static");
    }
    finish(')');
  }

  /* o->index may be an identifier or a local variable, in the latter case
   * need to ensure that it is only declared once in the first element of the
   * for loop */
  ++inAssign;
  auto param = dynamic_cast<LocalVariable*>(o->index);
  if (param) {
    Identifier<LocalVariable> ref(param->name, param->loc, param);
    start("for (" << param << " = " << o->from << "; ");
    middle(&ref << " <= " << o->to << "; ");
    finish("++" << &ref << ") {");
  } else {
    start("for (" << o->index << " = " << o->from << "; ");
    middle(o->index << " <= " << o->to << "; ");
    finish("++" << o->index << ") {");
  }
  --inAssign;
  in();
  *this << o->braces->strip();
  out();
  line("}");

  if (o->has(PARALLEL)) {
    out();
    line("}");
  }
}

void bi::CppBaseGenerator::visit(const While* o) {
  genTraceLine(o->loc->firstLine);
  start("while (");
  auto cond = o->cond->strip();
  if (cond->type->isClass()) {
    middle("*(");
  }
  middle(cond);
  if (cond->type->isClass()) {
    middle(')');
  }
  finish(") {");
  in();
  *this << o->braces->strip();
  out();
  line("}");
}

void bi::CppBaseGenerator::visit(const DoWhile* o) {
  line("do {");
  in();
  *this << o->braces->strip();
  out();
  genTraceLine(o->loc->lastLine);
  line("} while (" << o->cond->strip() << ");");
}

void bi::CppBaseGenerator::visit(const Assert* o) {
  genTraceLine(o->loc->firstLine);
  line("libbirch_assert_(" << o->cond->strip() << ");");
}

void bi::CppBaseGenerator::visit(const Return* o) {
  genTraceLine(o->loc->firstLine);
  line("return " << o->single << ';');
}

void bi::CppBaseGenerator::visit(const Yield* o) {
  assert(false);  // should be in CppFiberGenerator
}

void bi::CppBaseGenerator::visit(const Raw* o) {
  if ((header && *o->name == "hpp") || (!header && *o->name == "cpp")) {
    *this << escape_unicode(o->raw);
    if (!std::isspace(o->raw.back())) {
      *this << ' ';
    }
  }
}

void bi::CppBaseGenerator::visit(const StatementList* o) {
  middle(o->head << o->tail);
}

void bi::CppBaseGenerator::visit(const EmptyType* o) {
  middle("void");
}

void bi::CppBaseGenerator::visit(const ArrayType* o) {
  middle("libbirch::DefaultArray<" << o->single << ',' << o->depth() << '>');
}

void bi::CppBaseGenerator::visit(const TupleType* o) {
  middle("std::tuple<" << o->single << '>');
}
void bi::CppBaseGenerator::visit(const FunctionType* o) {
  middle("std::function<" << o->returnType << "(libbirch::Label* context_");
  if (!o->params->isEmpty()) {
    middle(", " << o->params);
  }
  middle(")>");
}

void bi::CppBaseGenerator::visit(const FiberType* o) {
  middle("libbirch::Fiber<" << o->single << '>');
}

void bi::CppBaseGenerator::visit(const OptionalType* o) {
  middle("libbirch::Optional<" << o->single << '>');
}

void bi::CppBaseGenerator::visit(const ClassType* o) {
  int inPointer1 = inPointer;
  if (!inPointer1) {
    if (o->weak) {
      middle("libbirch::Weak<");
    } else {
      middle("libbirch::Shared<");
    }
  } else {
    --inPointer;
  }
  middle("bi::type::" << o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
  if (!inPointer1) {
    middle('>');
  }
}

void bi::CppBaseGenerator::visit(const BasicType* o) {
  middle("bi::type::" << o->name);
}

void bi::CppBaseGenerator::visit(const GenericType* o) {
  if (o->target->type->isEmpty()) {
    middle(o->name);
  } else {
    middle(o->target->type);
  }
}

void bi::CppBaseGenerator::visit(const MemberType* o) {
  middle(o->right);
}

void bi::CppBaseGenerator::visit(const UnknownType* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const TypeList* o) {
  middle(o->head << ", " << o->tail);
}

void bi::CppBaseGenerator::genTraceFunction(const std::string& name,
    const Location* loc) {
  start("libbirch_function_(\"" << name << "\", \"");
  finish(loc->file->path << "\", " << loc->firstLine << ");");
}

void bi::CppBaseGenerator::genTraceLine(const int line) {
  line("libbirch_line_(" << line << ");");
}

void bi::CppBaseGenerator::genArgs(const Call* o) {
  middle('(');
  if (!o->single->isMember()) {
    middle("context_");
    if (!o->args->isEmpty()) {
      middle(", ");
    }
  }
  auto iter1 = o->args->begin();
  auto end1 = o->args->end();
  auto iter2 = o->callType->params->begin();
  auto end2 = o->callType->params->end();
  while (iter1 != end1 && iter2 != end2) {
    if (iter1 != o->args->begin()) {
      middle(", ");
    }
    genArg(*iter1, *iter2);
    ++iter1;
    ++iter2;
  }
  middle(')');
}

void bi::CppBaseGenerator::genLeftArg(const BinaryCall* o) {
  genArg(o->args->getLeft(), o->callType->params->getLeft());
}

void bi::CppBaseGenerator::genRightArg(const BinaryCall* o) {
  genArg(o->args->getRight(), o->callType->params->getRight());
}

void bi::CppBaseGenerator::genSingleArg(const UnaryCall* o) {
  genArg(o->args, o->callType->params);
}

void bi::CppBaseGenerator::genArg(const Expression* arg, const Type* type) {
  /* Birch and C++ resolve overloads differently, explicit casting avoids
   * situations where Birch considers a call unambiguous, whereas C++ does
   * not */
  auto isThis = dynamic_cast<const This*>(arg);
  auto isSuper = dynamic_cast<const Super*>(arg);
  auto isSequence = dynamic_cast<const Sequence*>(arg);
  if (!arg->type->equals(*type) || isThis || isSuper || isSequence) {
    middle(type->canonical() << '(' << arg << ')');
  } else {
    middle(arg);
  }
}
