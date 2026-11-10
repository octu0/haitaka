UNAME_S := $(shell uname -s)
SWIFT ?= swift
DATA ?= /path/to/manifest.jsonl
ENGLISH_DICT ?= /path/to/cmudict.dict
EPOCHS ?= 9
WEIGHTS ?= .tmp/weights.safetensors

ifeq ($(UNAME_S),Darwin)
SCHEME_PKG := haitaka-Package
SCHEME_TRAIN := train
DESTINATION := platform=macOS
CONFIG := Release
PRODUCTS = $$(xcodebuild -scheme $(SCHEME_TRAIN) -configuration $(CONFIG) -showBuildSettings 2>/dev/null | awk '/ BUILT_PRODUCTS_DIR =/{print $$3; exit}')
else
export PATH := /usr/local/cuda/bin:$(PATH)
export CUDA_ARCH ?= sm_86
export MLX_CCCL_DIR ?= $(HOME)/.local/share/cccl/include
export MLX_CUDA_INCLUDE_PATHS ?= $(HOME)/.local/share/cudnn-frontend/include:$(HOME)/.local/share/cutlass/include
export NVCC_PREPEND_FLAGS ?= -allow-unsupported-compiler --extended-lambda
export NVIDIA_TF32_OVERRIDE ?= 0
endif

.PHONY: all build build-train test train eval clean help

all: build test

ifeq ($(UNAME_S),Darwin)
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
else
build:
	$(SWIFT) build -c release

build-train:
	$(SWIFT) build -c release --product train

test:
	$(SWIFT) test

train:
	$(SWIFT) run -c release train -d $(DATA) --english-dict $(ENGLISH_DICT) -e $(EPOCHS) --export-weights $(WEIGHTS)

eval:
	$(SWIFT) run -c release train -d $(DATA) --english-dict $(ENGLISH_DICT) -e 0 --import-weights $(WEIGHTS)

clean:
	$(SWIFT) package clean
endif

help:
	@echo "make build        Build every target (Release)"
	@echo "make build-train  Build the train tool (Release)"
	@echo "make test         Run the test suite"
	@echo "make train DATA=<manifest.jsonl> ENGLISH_DICT=<cmudict> [EPOCHS=9 WEIGHTS=...]"
	@echo "make eval  DATA=<manifest.jsonl> ENGLISH_DICT=<cmudict> WEIGHTS=<weights.safetensors>"
	@echo "make clean        Remove build products"
