/**
 * @file
 */
#include "bi/io/cpp/CppBaseGenerator.hpp"

#include "bi/io/cpp/CppClassGenerator.hpp"
#include "bi/io/cpp/CppResumeGenerator.hpp"
#include "bi/primitive/encode.hpp"

#include "boost/algorithm/string.hpp"

bi::CppBaseGenerator::CppBaseGenerator(std::ostream& base, const int level,
    const bool header, const bool generic) :
    indentable_ostream(base, level),
    header(header),
    generic(generic),
    inAssign(0),
    inConstructor(0),
    inLambda(0),
    inMember(0),
    inSequence(0) {
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
  middle("bi::type::Integer(" << o->str << ')');
}

void bi::CppBaseGenerator::visit(const Literal<double>* o) {
  middle(o->str);
}

void bi::CppBaseGenerator::visit(const Literal<const char*>* o) {
  middle("bi::type::String(" << o->str << ')');
}

void bi::CppBaseGenerator::visit(const Parentheses* o) {
  auto stripped = o->strip();
  if (stripped->isTuple()) {
    if (inAssign) {
      --inAssign;
      middle("libbirch::tie(");
      for (auto iter = stripped->begin(); iter != stripped->end(); ++iter) {
        if (iter != stripped->begin()) {
          middle(", ");
        }
        if ((*iter)->isTuple() || (*iter)->isSlice()) {
          ++inAssign;
        }
        middle(*iter);
      }
      middle(')');
    } else {
      middle("libbirch::make_tuple(");
      for (auto iter = stripped->begin(); iter != stripped->end(); ++iter) {
        if (iter != stripped->begin()) {
          middle(", ");
        }
        middle("libbirch::canonical(" << *iter << ')');
      }
      middle(')');
    }
  } else {
    middle('(' << stripped << ')');
  }
}

void bi::CppBaseGenerator::visit(const Sequence* o) {
  if (o->single->isEmpty()) {
    middle("libbirch::nil");
  } else if (!inSequence) {
    middle("libbirch::make_array(");
    ++inSequence;
    middle("{ " << o->single << " }");
    --inSequence;
    middle(')');
  } else {
    middle("{ " << o->single << " }");
  }
}

void bi::CppBaseGenerator::visit(const Cast* o) {
  middle("libbirch::cast<" << o->returnType << ">(" << o->single << ')');
}

void bi::CppBaseGenerator::visit(const Call* o) {
  middle(o->single << '(' << o->args << ')');
}

void bi::CppBaseGenerator::visit(const BinaryCall* o) {
  if (isTranslatable(o->name->str())) {
    middle(o->left << ' ' << o->name->str() << ' ' << o->right);
  } else {
    middle(o->name << '(' << o->left << ", " << o->right << ')');
  }
}

void bi::CppBaseGenerator::visit(const UnaryCall* o) {
  if (isTranslatable(o->name->str())) {
    middle(o->name->str() << o->single);
  } else {
    middle(o->name << '(' << o->single << ')');
  }
}

void bi::CppBaseGenerator::visit(const Assign* o) {
  if (o->left->isSlice() || o->left->isTuple()) {
    ++inAssign;
  }
  middle(o->left << " = " << o->right);
}

void bi::CppBaseGenerator::visit(const Slice* o) {
  middle(o->single << '.');
  if (inAssign) {
    --inAssign;
    middle("get");
  } else {
    middle("pull");
  }
  middle("(libbirch::make_slice(" << o->brackets << "))");
}

void bi::CppBaseGenerator::visit(const Query* o) {
  middle(o->single << ".query()");
}

void bi::CppBaseGenerator::visit(const Get* o) {
  middle(o->single << ".get()");
}

void bi::CppBaseGenerator::visit(const Spin* o) {
  middle(o->single << ".spin()");
}

void bi::CppBaseGenerator::visit(const LambdaFunction* o) {
  finish("[=](" << o->params << ") {");
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
    /* don't have access to the `self` variable here, but because we're
     * constructing, so the object cannot possibly be frozen, `this`
     * suffices */
    if (leftThis) {
      middle("this->");
    } else if (leftSuper) {
      middle("this->super_type_::");
    }
  } else {
    if (leftThis) {
      middle("self()->");
    } else if (leftSuper) {
      middle("self()->super_type_::");
    } else {
      middle(o->left);
      ///@todo Restore read-only optimization
      //auto rightVar = dynamic_cast<const NamedExpression*>(o->right);
      //if (!inAssign && rightVar) {
        /* optimization: just reading a value, so no need to copy-on-write the
         * owning object */
      //  middle(".pull()");
      //}
      middle("->");
    }
  }
  ++inMember;
  middle(o->right);
  --inMember;
}

void bi::CppBaseGenerator::visit(const This* o) {
  if (inConstructor) {
    middle("this");
  } else {
    middle("self()");
  }
}

void bi::CppBaseGenerator::visit(const Super* o) {
  assert(false);  // should be handled by visit(Member*)
}

void bi::CppBaseGenerator::visit(const Global* o) {
  middle(o->single);
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

void bi::CppBaseGenerator::visit(const NamedExpression* o) {
  if (o->isGlobal()) {
    middle("bi::" << o->name);
    if (o->category == GLOBAL_VARIABLE) {
      /* global variables generated as functions */
      middle("()");
    }
  } else if (o->isMember()) {
    if (!inMember && !inConstructor) {
      middle("self()->");
    }
    middle(o->name);
  } else {
    middle(o->name);
  }
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
}

void bi::CppBaseGenerator::visit(const File* o) {
  *this << o->root;
}

void bi::CppBaseGenerator::visit(const GlobalVariable* o) {
  /* C++ does not guarantee static initialization order across compilation
   * units. Global variables are therefore used through accessor functions
   * that initialize their values on first use. */
  genSourceLine(o->loc);
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
    genSourceLine(o->loc);
    start("static " << o->type << " result");
    genInit(o);
    finish(';');
    genSourceLine(o->loc);
    line("return result;");
    out();
    line("}\n");
  }
}

void bi::CppBaseGenerator::visit(const MemberVariable* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const LocalVariable* o) {
  genTraceLine(o->loc);
  if (o->has(AUTO)) {
    start("auto " << o->name);
  } else {
    start(o->type << ' ' << o->name);
  }
  genInit(o);
  finish(';');
}

void bi::CppBaseGenerator::visit(const Function* o) {
  if ((generic || !o->isGeneric()) && !o->braces->isEmpty()) {
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
      genTraceFunction(o->name->str(), o->loc);
      *this << o->braces->strip();
      out();
      line("}\n");
    }
  }
}

void bi::CppBaseGenerator::visit(const Fiber* o) {
  if (generic || !o->isGeneric()) {
    /* initialization function */
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
      genTraceLine(o->loc);
      line("yield_" << o->name << '_' << o->number << "_0_();");
      out();
      line("}\n");
    }

    /* start function */
    CppResumeGenerator auxResume(nullptr, o, base, level, header);
    auxResume << o->start;

    /* resume functions */
    Gatherer<Yield> yields;
    o->accept(&yields);
    for (auto yield : yields) {
      if (yield->resume) {
        CppResumeGenerator auxResume(nullptr, o, base, level, header);
        auxResume << yield->resume;
      }
    }
  }
}

void bi::CppBaseGenerator::visit(const MemberFunction* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const MemberFiber* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const Program* o) {
  genSourceLine(o->loc);
  if (header) {
    line("extern \"C\" int " << o->name << "(int, char**);");
  } else {
    line("int bi::" << o->name << "(int argc_, char** argv_) {");
    in();
    genTraceFunction(o->name->str(), o->loc);

    /* handle program options */
    if (o->params->width() > 0) {
      /* option variables */
      for (auto iter = o->params->begin(); iter != o->params->end(); ++iter) {
        auto param = dynamic_cast<const Parameter*>(*iter);
        assert(param);
        genSourceLine(o->loc);
        start(param->type << ' ' << param->name);
        if (!param->value->isEmpty()) {
          middle(" = " << param->value);
        } else if (param->type->isClass()) {
          middle(" = libbirch::make_pointer<" << param->type << ">()");
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
      genSourceLine(o->loc);
      line("int c_, option_index_;");
      genSourceLine(o->loc);
      line("option long_options_[] = {");
      in();
      for (auto param : *o->params) {
        auto name = dynamic_cast<const Parameter*>(param)->name;
        std::string flag = internalise(name->str()) + "FLAG_";

        std::string option = name->str();
        boost::replace_all(option, "_", "-");

        genSourceLine(o->loc);
        start("{\"");
        middle(option << "\", required_argument, 0, " << flag);
        finish(" },");
      }
      genSourceLine(o->loc);
      line("{0, 0, 0, 0}");
      out();
      genSourceLine(o->loc);
      line("};");

      /* short options */
      genSourceLine(o->loc);
      line("const char* short_options_ = \":\";");

      /* handle error reporting ourselves (worth commenting this out if
       * debugging issues with command-line parsing) */
      genSourceLine(o->loc);
      line("::opterr = 0;");

      /* read in options with getopt_long */
      genSourceLine(o->loc);
      start("c_ = ::getopt_long_only(argc_, argv_, short_options_, ");
      finish("long_options_, &option_index_);");
      genSourceLine(o->loc);
      line("while (c_ != -1) {");
      in();
      genSourceLine(o->loc);
      line("switch (c_) {");
      in();

      for (auto param : *o->params) {
        auto p = dynamic_cast<const Parameter*>(param);
        auto name = p->name;
        std::string flag = internalise(name->str()) + "FLAG_";

        genSourceLine(p->loc);
        line("case " << flag << ':');
        in();
        genSourceLine(p->loc);
        line("libbirch_error_msg_(::optarg, \"option --\" << long_options_[::optopt].name << \" requires a value.\");");
        genSourceLine(p->loc);
        if (p->type->unwrap()->isBasic()) {
          auto type = dynamic_cast<Named*>(p->type->unwrap());
          assert(type);
          start(name << " = bi::" << type->name);
          finish("(std::string(::optarg));");
        } else {
          line(name << " = std::string(::optarg);");
        }
        line("break;");
        out();
      }

      genSourceLine(o->loc);
      line("case '?':");
      in();
      genSourceLine(o->loc);
      line("libbirch_error_msg_(false, \"option \" << argv_[::optind - 1] << \" unrecognized.\");");
      out();

      genSourceLine(o->loc);
      line("case ':':");
      in();
      genSourceLine(o->loc);
      line("libbirch_error_msg_(false, \"option --\" << long_options_[::optopt].name << \" requires a value.\");");
      out();

      genSourceLine(o->loc);
      line("default:");
      in();
      genSourceLine(o->loc);
      line("libbirch_error_msg_(false, std::string(\"unknown error parsing command-line options.\"));");
      out();

      out();
      line('}');
      genSourceLine(o->loc);
      start("c_ = ::getopt_long_only(argc_, argv_, short_options_, ");
      finish("long_options_, &option_index_);");
      out();
      line("}\n");
    }

    /* seed random number generator with random entropy */
    genTraceLine(o->loc);
    line("bi::seed();\n");

    /* body of program */
    *this << o->braces->strip();

    genTraceLine(o->loc);
    line("return 0;");
    out();
    line("}\n");
  }
}

void bi::CppBaseGenerator::visit(const BinaryOperator* o) {
  if (!o->braces->isEmpty()) {
    genSourceLine(o->loc);
    start(o->returnType << ' ');
    if (!header) {
      middle("bi::");
    }
    if (isTranslatable(o->name->str())) {
      middle("operator" << o->name->str());
    } else {
      middle(o->name);
    }
    middle('(' << o->left << ", " << o->right << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceFunction(o->name->str(), o->loc);
      *this << o->braces->strip();
      out();
      finish("}\n");
    }
  }
}

void bi::CppBaseGenerator::visit(const UnaryOperator* o) {
  if (!o->braces->isEmpty()) {
    genSourceLine(o->loc);
    start(o->returnType << ' ');
    if (!header) {
      middle("bi::");
    }
    if (isTranslatable(o->name->str())) {
      middle("operator" << o->name->str());
    } else {
      middle(o->name);
    }
    middle('(' << o->single << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceFunction(o->name->str(), o->loc);
      *this << o->braces->strip();
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
  //
}

void bi::CppBaseGenerator::visit(const Class* o) {
  if (generic || !o->isGeneric()) {
    CppClassGenerator auxClass(base, level, header);
    auxClass << o;
  }
}

void bi::CppBaseGenerator::visit(const Generic* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const Braces* o) {
  line('{');
  in();
  *this << o->single;
  out();
  line('}');
}

void bi::CppBaseGenerator::visit(const Assume* o) {
  assert(false);  // should have been replaced by Transformer
}

void bi::CppBaseGenerator::visit(const ExpressionStatement* o) {
  genTraceLine(o->loc);
  line(o->single << ';');
}

void bi::CppBaseGenerator::visit(const If* o) {
  genTraceLine(o->loc);
  line("if (" << o->cond->strip() << ") {");
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
  auto index = dynamic_cast<const LocalVariable*>(o->index);
  assert(index);

  genTraceLine(o->loc);
  start("for (auto " << index->name << " = " << o->from << "; ");
  finish(index->name << " <= " << o->to << "; ++" << index->name << ") {");
  in();
  *this << o->braces->strip();
  out();
  line("}");
}

void bi::CppBaseGenerator::visit(const Parallel* o) {
  auto index = dynamic_cast<const LocalVariable*>(o->index);
  assert(index);

  genTraceLine(o->loc);
  line("#pragma omp parallel");
  line("{");
  in();
  genTraceFunction("<parallel for>", o->loc);
  start("#pragma omp for schedule(");
  if (o->has(DYNAMIC)) {
    middle("guided");
  } else {
    middle("static");
  }
  finish(')');
  start("for (auto " << index->name << " = " << o->from << "; ");
  finish(index->name << " <= " << o->to << "; ++" << index->name << ") {");
  in();
  *this << o->braces->strip();
  out();
  line("}");
  out();
  line("}");
}

void bi::CppBaseGenerator::visit(const While* o) {
  genTraceLine(o->loc);
  line("while (" << o->cond->strip() << ") {");
  in();
  *this << o->braces->strip();
  out();
  line("}");
}

void bi::CppBaseGenerator::visit(const DoWhile* o) {
  genTraceLine(o->loc);
  line("do {");
  in();
  *this << o->braces->strip();
  out();
  line("} while (" << o->cond->strip() << ");");
}

void bi::CppBaseGenerator::visit(const Assert* o) {
  genTraceLine(o->loc);
  line("libbirch_assert_(" << o->cond->strip() << ");");
}

void bi::CppBaseGenerator::visit(const Return* o) {
  genTraceLine(o->loc);
  line("return " << o->single << ';');
}

void bi::CppBaseGenerator::visit(const Yield* o) {
  assert(false);  // should be in CppResumeGenerator
}

void bi::CppBaseGenerator::visit(const Raw* o) {
  if ((header && *o->name == "hpp") || (!header && *o->name == "cpp")) {
    if (!header) {
      genSourceLine(o->loc);
    }
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
  middle("libbirch::Tuple<" << o->single << '>');
}

void bi::CppBaseGenerator::visit(const FunctionType* o) {
  middle("std::function<" << o->returnType << '(' << o->params << ")>");
}

void bi::CppBaseGenerator::visit(const FiberType* o) {
  middle("libbirch::Fiber<" << o->yieldType << ',' << o->returnType << '>');
}

void bi::CppBaseGenerator::visit(const OptionalType* o) {
  middle("libbirch::Optional<" << o->single << '>');
}

void bi::CppBaseGenerator::visit(const MemberType* o) {
  middle(o->right);
}

void bi::CppBaseGenerator::visit(const NamedType* o) {
  if (o->isClass()) {
    if (o->weak) {
      middle("libbirch::Lazy<libbirch::Weak<");
    } else {
      middle("libbirch::Lazy<libbirch::Shared<");
    }
    middle("bi::type::" << o->name);
    if (!o->typeArgs->isEmpty()) {
      middle('<' << o->typeArgs << '>');
    }
    middle(">>");
  } else if (o->isBasic()) {
    middle("bi::type::" << o->name);
    if (!o->typeArgs->isEmpty()) {
      middle('<' << o->typeArgs << '>');
    }
  } else if (o->isGeneric()) {
    middle(o->name);
    if (!o->typeArgs->isEmpty()) {
      middle('<' << o->typeArgs << '>');
    }
  } else {
    assert(false);
  }
}

void bi::CppBaseGenerator::visit(const TypeList* o) {
  middle(o->head << ", " << o->tail);
}

void bi::CppBaseGenerator::genTraceFunction(const std::string& name,
    const Location* loc) {
  genSourceLine(loc);
  start("libbirch_function_(\"" << name << "\", \"");
  finish(loc->file->path << "\", " << loc->firstLine << ");");
}

void bi::CppBaseGenerator::genTraceLine(const Location* loc) {
  genSourceLine(loc);
  line("libbirch_line_(" << loc->firstLine << ");");
  genSourceLine(loc);
}

void bi::CppBaseGenerator::genSourceLine(const Location* loc) {
  line("//#line " << loc->firstLine << " \"" << loc->file->path << "\"");
}
