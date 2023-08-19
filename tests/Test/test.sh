#!/bin/bash
set -eov pipefail

N1=1000   # for cdf tests
N2=100    # for gradient tests
N3=10000  # for pdf tests
N4=10000  # for normalizing constant tests
N5=10000  # for conjugacy tests
S=10      # for pdf tsts

eval "`grep -r "program test_basic_" src     | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1/"                         | sort`"
eval "`grep -r "program test_cdf_" src       | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N1/"                  | sort`"
eval "`grep -r "program test_grad_" src      | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N2 --backward false/" | sort`"
eval "`grep -r "program test_grad_" src      | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N2 --backward true/"  | sort`"
eval "`grep -r "program test_pdf_" src       | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N3 -S $S --lazy false/"     | sort`"
eval "`grep -r "program test_pdf_" src       | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N3 -S $S --lazy true/"      | sort`"
eval "`grep -r "program test_z_" src         | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N4 --lazy false/"     | sort`"
eval "`grep -r "program test_z_" src         | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N4 --lazy true/"      | sort`"
eval "`grep -r "program test_conjugacy_" src | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N5 --lazy false/"     | sort`"
eval "`grep -r "program test_conjugacy_" src | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N5 --lazy true/"      | sort`"
