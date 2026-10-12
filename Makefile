SCHEME_PKG := haitaka-Package
SCHEME_TRAIN := train
DESTINATION := platform=macOS
CONFIG := Release
PRODUCTS = $$(xcodebuild -scheme $(SCHEME_TRAIN) -configuration $(CONFIG) -showBuildSettings 2>/dev/null | awk '/ BUILT_PRODUCTS_DIR =/{print $$3; exit}')

DATA ?= /path/to/manifest.jsonl
ENGLISH_DICT ?= /path/to/cmudict.dict
EPOCHS ?= 9
WEIGHTS ?= .tmp/weights.json

.PHONY: all build build-train test train eval clean help

all: build test

build:
	xcodebuild build -scheme $(SCHEME_PKG) -destination '$(DESTINATION)' -configuration $(CONFIG)

build-train:
	xcodebuild build -scheme $(SCHEME_TRAIN) -destination '$(DESTINATION)' -configuration $(CONFIG)

test:
	xcodebuild test -scheme $(SCHEME_PKG) -destination '$(DESTINATION)' -configuration $(CONFIG) ENABLE_TESTABILITY=YES

train: build-train
	$(PRODUCTS)/train -d $(DATA) --english-dict $(ENGLISH_DICT) -e $(EPOCHS) --export-weights $(WEIGHTS)

eval: build-train
	$(PRODUCTS)/train -d $(DATA) --english-dict $(ENGLISH_DICT) -e 0 --import-weights $(WEIGHTS)

clean:
	xcodebuild clean -scheme $(SCHEME_PKG) -destination '$(DESTINATION)' -configuration $(CONFIG)
	swift package clean

help:
	@echo "make build        Build every target (Release)"
	@echo "make build-train  Build the train tool (Release)"
	@echo "make test         Run the test suite (Release)"
	@echo "make train DATA=<manifest.jsonl> ENGLISH_DICT=<cmudict> [EPOCHS=9 WEIGHTS=...]"
	@echo "make eval  DATA=<manifest.jsonl> ENGLISH_DICT=<cmudict> WEIGHTS=<weights.json>"
	@echo "make clean        Remove build products"
