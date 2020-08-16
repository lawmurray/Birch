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

docs: $(PACKAGES) driver

.PHONY: $(PACKAGES)
$(PACKAGES):
	mkdir -p docs/$@
	cd $@ && birch docs
	cp -R $@/docs/* docs/$@/.

.PHONY: driver
driver:
	mkdir -p docs/$@
	cp -R $@/docs/* docs/$@/.
