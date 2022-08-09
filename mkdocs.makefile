PACKAGES := \
	libraries/Cairo \
	libraries/SQLite \
	libraries/Standard \
        examples/Ecology \
	examples/LinearGaussian \
	examples/LinearRegression \
	examples/MixedGaussian \
	examples/MultiObjectTracking \
	examples/PoissonGaussian \
	examples/SIR \
	examples/VectorBorneDisease

docs: $(PACKAGES) membirch
	cp README.md docs/index.md

.PHONY: $(PACKAGES)
$(PACKAGES):
	mkdir -p docs/$@
	cd $@ && birch docs
	cp -R $@/docs/* docs/$@/.

.PHONY: membirch
membirch:
	mkdir -p docs/$@
	cd $@ && doxygen
	cp -R $@/docs/html/* docs/$@/.
