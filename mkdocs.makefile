PACKAGES := \
	libraries/Cairo \
	libraries/SQLite \
	libraries/Standard \
	examples/LinearGaussian \
	examples/LinearRegression \
	examples/MixedGaussian \
	examples/MultiObjectTracking \
	examples/PoissonGaussian \
	examples/SIR \
	examples/VectorBorneDisease

docs: $(PACKAGES) libbirch
	cp README.md docs/index.md

.PHONY: $(PACKAGES)
$(PACKAGES):
	mkdir -p docs/$@
	cd $@ && birch docs
	cp -R $@/docs/* docs/$@/.

.PHONY: libbirch
libbirch:
	mkdir -p docs/$@
	cd $@ && doxygen
	cp -R $@/docs/html/* docs/$@/.
