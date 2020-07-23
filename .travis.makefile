TEST_FLAGS = --enable-coverage
BENCH_FLAGS = --disable-debug
NJOBS = 2

all: compilers libraries tests benchmarks

compilers: test_compiler_Birch

libraries: \
	test_library_Standard \
	test_library_Cairo \
	test_library_SQLite \
	test_library_Test

tests: \
	test_test_TestBasic \
	test_test_TestCdf \
	test_test_TestConjugacy \
	test_test_TestPdf \
	test_example_LinearGaussian \
	test_example_LinearRegression \
	test_example_MixedGaussian \
	test_example_MultiObjectTracking \
	test_example_PoissonGaussian \
	test_example_SIR \
	test_example_VectorBorneDisease

benchmarks: \
	bench_example_LinearGaussian \
	bench_example_LinearRegression \
	bench_example_MixedGaussian \
	bench_example_MultiObjectTracking \
	bench_example_PoissonGaussian \
	bench_example_SIR \
	bench_example_VectorBorneDisease

test_compiler_Birch:
	cd compilers/Birch && \
	./autogen.sh && \
	mkdir -p build && \
	cd build && \
	../configure --config-cache CFLAGS="-Wall -Wno-overloaded-virtual -g -O0 --coverage" CXXFLAGS="-Wall -Wno-overloaded-virtual -g -O0 --coverage" INSTALL="install -p" && \
	make -j $(NJOBS) && \
	sudo make install

bench_compiler_Birch: test_compiler_birch

test_library_Standard: test_compiler_Birch
	cd libraries/Birch.Standard && \
	birch build $(TEST_FLAGS) && \
	sudo birch install $(TEST_FLAGS)

bench_library_Standard: bench_compiler_Birch
	cd libraries/Birch.Standard && \
	birch build $(BENCH_FLAGS) && \
	sudo birch install $(BENCH_FLAGS)

test_library_Cairo: test_library_Standard
	cd libraries/Birch.Cairo && \
	birch build $(TEST_FLAGS) && \
	sudo birch install $(TEST_FLAGS)

bench_library_Cairo: bench_library_Standard
	cd libraries/Birch.Cairo && \
	birch build $(BENCH_FLAGS) && \
	sudo birch install $(BENCH_FLAGS)

test_library_SQLite: test_library_Standard
	cd libraries/Birch.SQLite && \
	birch build $(TEST_FLAGS) && \
	sudo birch install $(TEST_FLAGS)

bench_library_SQLite: bench_library_Standard
	cd libraries/Birch.SQLite && \
	birch build $(BENCH_FLAGS) && \
	sudo birch install $(BENCH_FLAGS)

test_library_Test: test_library_Standard
	cd libraries/Birch.Test && \
	birch build $(TEST_FLAGS) && \
	sudo birch install $(TEST_FLAGS)

bench_library_Test: bench_library_Standard
	cd libraries/Birch.Test && \
	birch build $(BENCH_FLAGS) && \
	sudo birch install $(BENCH_FLAGS)

test_test_TestBasic: test_library_Standard test_library_Test
	cd tests/TestBasic && \
	birch build $(TEST_FLAGS) && \
	sudo birch install $(TEST_FLAGS) && \
	./run

test_test_TestCdf: test_library_Standard test_library_Test
	cd tests/TestCdf && \
	birch build $(TEST_FLAGS) && \
	sudo birch install $(TEST_FLAGS) && \
	./run

test_test_TestConjugacy: test_library_Standard test_library_Test
	cd tests/TestConjugacy && \
	birch build $(TEST_FLAGS) && \
	sudo birch install $(TEST_FLAGS) && \
	./run

test_test_TestPdf: test_library_Standard test_library_Test
	cd tests/TestPdf && \
	birch build $(TEST_FLAGS) && \
	sudo birch install $(TEST_FLAGS) && \
	./run

test_example_LinearGaussian: test_library_Standard
	cd examples/LinearGaussian && \
	birch build $(TEST_FLAGS) && \
	sudo birch install $(TEST_FLAGS) && \
	birch sample --config config/linear_gaussian.json

bench_example_LinearGaussian: bench_library_Standard
	cd examples/LinearGaussian && \
	birch build $(BENCH_FLAGS) && \
	sudo birch install $(BENCH_FLAGS) && \
	hyperfine --warmup 1 --export-csv LinearGaussian.csv birch sample --config config/linear_gaussian.json

test_example_LinearRegression: test_library_Standard
	cd examples/LinearRegression && \
	birch build $(TEST_FLAGS) && \
	sudo birch install $(TEST_FLAGS) && \
	birch sample --config config/linear_regression.json

bench_example_LinearRegression: bench_library_Standard
	cd examples/LinearRegression && \
	birch build $(BENCH_FLAGS) && \
	sudo birch install $(BENCH_FLAGS) && \
	hyperfine --warmup 1 --export-csv $@.csv 'birch sample --config config/linear_regression.json'

test_example_MixedGaussian: test_library_Standard
	cd examples/MixedGaussian && \
	birch build $(TEST_FLAGS) && \
	sudo birch install $(TEST_FLAGS) && \
	birch sample --config config/mixed_gaussian.json

bench_example_MixedGaussian: bench_library_Standard
	cd examples/MixedGaussian && \
	birch build $(BENCH_FLAGS) && \
	sudo birch install $(BENCH_FLAGS) && \
	hyperfine --warmup 1 --export-csv $@.csv 'birch sample --config config/mixed_gaussian.json'

test_example_MultiObjectTracking: test_library_Standard test_library_Cairo
	cd examples/MultiObjectTracking && \
	birch build $(TEST_FLAGS) && \
	sudo birch install $(TEST_FLAGS) && \
	birch filter --config config/filter.json --output ""

bench_example_MultiObjectTracking: bench_library_Standard bench_library_Cairo
	cd examples/MultiObjectTracking && \
	birch build $(BENCH_FLAGS) && \
	sudo birch install $(BENCH_FLAGS) && \
	hyperfine --warmup 1 --export-csv $@.csv 'birch filter --config config/filter.json --output ""'

test_example_PoissonGaussian: test_library_Standard
	cd examples/PoissonGaussian && \
	birch build $(TEST_FLAGS) && \
	sudo birch install $(TEST_FLAGS) && \
	birch sample --config config/poisson_gaussian.json

bench_example_PoissonGaussian: bench_library_Standard
	cd examples/PoissonGaussian && \
	birch build $(BENCH_FLAGS) && \
	sudo birch install $(BENCH_FLAGS) && \
	hyperfine --warmup 1 --export-csv $@.csv 'birch sample --config config/poisson_gaussian.json'

test_example_SIR: test_library_Standard
	cd examples/SIR && \
	birch build $(TEST_FLAGS) && \
	sudo birch install $(TEST_FLAGS) && \
	birch sample --config config/sir.json

bench_example_SIR: bench_library_Standard
	cd examples/SIR && \
	birch build $(BENCH_FLAGS) && \
	sudo birch install $(BENCH_FLAGS) && \
	hyperfine --warmup 1 --export-csv $@.csv 'birch sample --config config/sir.json'

test_example_VectorBorneDisease: test_library_Standard
	cd examples/VectorBorneDisease && \
	birch build $(TEST_FLAGS) && \
	sudo birch install $(TEST_FLAGS) && \
	birch sample --config config/yap_dengue.json

bench_example_VectorBorneDisease: bench_library_Standard
	cd examples/VectorBorneDisease && \
	birch build $(BENCH_FLAGS) && \
	sudo birch install $(BENCH_FLAGS) && \
	hyperfine --warmup 1 --export-csv $@.csv 'birch sample --config config/yap_dengue.json'
