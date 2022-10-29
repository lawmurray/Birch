PACKAGES := \
	libraries/Cairo \
	libraries/SQLite \
	libraries/Standard

docs: $(PACKAGES)
	cp README.md docs/index.md

.PHONY: $(PACKAGES)
$(PACKAGES):
	mkdir -p docs/$@
	cd $@ && birch docs
	cp -R $@/docs/* docs/$@/.
