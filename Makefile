# EcoSort - developer task runner.
#
# Run `make` (or `make help`) for the list of targets.
# Requires Docker Compose v2 (`docker compose`, not `docker-compose`).

SHELL := /bin/sh
.DEFAULT_GOAL := help

COMPOSE      := docker compose
COMPOSE_PROD := docker compose -f docker-compose.yml -f docker-compose.prod.yml

PYTHON       ?= python3
VENV         := ml/.venv
PY           := $(VENV)/bin/python
VENV_STAMP   := $(VENV)/.ecosort-requirements-installed

# Override on the command line, e.g. `make train TRAIN_ARGS="--epochs 30"`.
TRAIN_ARGS   ?=
# --smoke-test synthesises a tiny random dataset, so this needs no images on disk
# and exports to ml/artifacts/smoke-export rather than touching models/custom.
SMOKE_ARGS   ?= --smoke-test
PREPARE_ARGS ?=
# The test split, not val: val was consumed by ModelCheckpoint and EarlyStopping during
# training, so it is no longer held out and its accuracy is optimistic.
EVAL_ARGS    ?= --split test
# `make inspect-data DIR=...` explores one downloaded dataset; `make ingest-data
# INGEST_ARGS="--source name=dir ..."` merges them into ml/source.
DIR          ?=
INGEST_ARGS  ?=
# `make dataset` - e.g. DATASET_ARGS="--cap 2000 --garbage12 ~/Downloads/garbage_classification".
DATASET_ARGS ?=
# The recommended NVIDIA recipe (ml/README.md, "Training on an NVIDIA GPU"). TRAIN_ARGS is
# appended after it and argparse keeps the last value, so `make train-gpu TRAIN_ARGS="--backbone
# efficientnetv2b2"` swaps one setting without restating the rest. GPU_PRECISION= turns mixed
# precision off for cards without tensor cores (pre-2017, compute capability < 7.0).
GPU_PRECISION  ?= --mixed-precision
GPU_TRAIN_ARGS ?= --backbone efficientnetv2b0 --batch-size 64 --epochs 8 --fine-tune-epochs 30 \
                  --mixup 0.2 --cutmix 1.0 --mix-prob 0.3 --quantize float16

# `make convert-model ARCH=InceptionV3` converts one pretrained keras.applications
# classifier; `make list-models` prints the ones that are supported. float16 is the
# default because it halves the browser download for no measurable accuracy loss.
ARCH         ?= InceptionResNetV2
QUANTIZE     ?= float16
CONVERT_ARGS ?=

.PHONY: help install fetch-models \
	up up-fg down restart logs logs-backend logs-frontend ps build rebuild \
	prod-up prod-down shell-backend shell-frontend \
	dev-backend dev-frontend test test-backend test-frontend lint \
	clean db-reset venv train train-smoke inspect-data ingest-data prepare-data evaluate doctor \
	convert-model list-models dataset venv-gpu gpu-check train-gpu

##@ General

help: ## Show this help
	@awk 'BEGIN { FS = ":.*##"; printf "\nEcoSort - make targets\n" } \
	     /^[a-zA-Z0-9_-]+:.*##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } \
	     /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@echo ""

# Any models/<dir>/ holding a model.json is a selectable engine, so doctor loops rather
# than checking two known names. The glob skips dot-directories on purpose: a conversion
# still writing into .<arch>.staging-<pid>/ is not an installed model yet.
doctor: ## Print tool versions and report which models are installed
	@echo "EcoSort environment"
	@echo "==================="
	@printf "docker          : "; docker --version 2>/dev/null || echo "NOT FOUND"
	@printf "docker compose  : "; $(COMPOSE) version --short 2>/dev/null || echo "NOT FOUND"
	@printf "node            : "; node --version 2>/dev/null || echo "NOT FOUND"
	@printf "npm             : "; npm --version 2>/dev/null || echo "NOT FOUND"
	@printf "python3         : "; $(PYTHON) --version 2>/dev/null || echo "NOT FOUND"
	@printf "ml venv         : "; if [ -x "$(PY)" ]; then $(PY) --version; else echo "not created (run 'make venv')"; fi
	@echo ""
	@echo "Models"
	@echo "======"
	@found=0; \
	 for manifest in models/*/model.json; do \
	   [ -f "$$manifest" ] || continue; \
	   dir=$${manifest%/model.json}; \
	   size=`du -sh "$$dir" 2>/dev/null | cut -f1`; \
	   case "$$dir" in \
	     models/custom)       what="trained classifier, preferred" ;; \
	     models/mobilenet_v2) what="pretrained ImageNet fallback" ;; \
	     *)                   what="converted ImageNet model" ;; \
	   esac; \
	   printf "%-20s: present (%s, %s)\n" "$$dir" "$$what" "$$size"; \
	   found=1; \
	 done; \
	 if [ $$found -eq 0 ]; then echo "none installed      : run 'make fetch-models'"; fi
	@if [ ! -f models/custom/model.json ]; then \
	   echo "models/custom       : MISSING  -> optional; run 'make train' to create it"; \
	 fi
	@if [ ! -f models/mobilenet_v2/model.json ]; then \
	   echo "models/mobilenet_v2 : MISSING  -> run 'make fetch-models'"; \
	 fi
	@if [ -f models/detectors/ssdlite_mobilenet_v2/model.json ]; then \
	   printf "%-20s: present (%s, %s)\n" "models/detectors" "Live scan object detector" \
	     "`du -sh models/detectors/ssdlite_mobilenet_v2 2>/dev/null | cut -f1`"; \
	 else \
	   echo "models/detectors    : MISSING  -> run 'make fetch-models' (needed by Live scan)"; \
	 fi

install: ## Install backend + frontend npm dependencies on the host
	@cd backend  && if [ -f package-lock.json ]; then npm ci; else npm install; fi
	@cd frontend && if [ -f package-lock.json ]; then npm ci; else npm install; fi

fetch-models: ## Download the pretrained MobileNetV2 fallback and the Live scan object detector into ./models
	node scripts/fetch-mobilenet.mjs
	node scripts/fetch-detector.mjs

##@ Docker (development stack)

up: ## Build and start the dev stack in the background
	$(COMPOSE) up --build -d
	@echo ""
	@echo "  Frontend : http://localhost:$${FRONTEND_PORT:-5173}"
	@echo "  API      : http://localhost:$${BACKEND_PORT:-4000}/api/health"
	@echo "  Logs     : make logs"

up-fg: ## Build and start the dev stack in the foreground (Ctrl-C to stop)
	$(COMPOSE) up --build

down: ## Stop the dev stack (keeps the database volume)
	$(COMPOSE) down

restart: ## Restart both services
	$(COMPOSE) restart

logs: ## Follow logs from all services
	$(COMPOSE) logs -f --tail=100

logs-backend: ## Follow backend logs
	$(COMPOSE) logs -f --tail=100 backend

logs-frontend: ## Follow frontend logs
	$(COMPOSE) logs -f --tail=100 frontend

ps: ## Show container status
	$(COMPOSE) ps

build: ## Build both images (using the layer cache)
	$(COMPOSE) build

rebuild: ## Build both images from scratch (no cache)
	$(COMPOSE) build --no-cache

shell-backend: ## Open a shell inside the running backend container
	$(COMPOSE) exec backend sh

shell-frontend: ## Open a shell inside the running frontend container
	$(COMPOSE) exec frontend sh

##@ Docker (production profile)

prod-up: ## Build and start the nginx production stack in the background
	$(COMPOSE_PROD) up --build -d
	@echo ""
	@echo "  App : http://localhost:$${FRONTEND_PROD_PORT:-8080}"

prod-down: ## Stop the production stack
	$(COMPOSE_PROD) down

##@ Local development (no Docker)

dev-backend: ## Run the Express API on the host with hot reload
	cd backend && npm run dev

dev-frontend: ## Run the Vite dev server on the host
	cd frontend && npm run dev

test: test-backend test-frontend ## Run every test suite

test-backend: ## Run the backend test suite
	cd backend && npm test

test-frontend: ## Run the frontend test suite
	cd frontend && npm test

lint: ## Lint both workspaces
	cd backend && npm run --if-present lint
	cd frontend && npm run --if-present lint

##@ Machine learning

venv: $(VENV_STAMP) ## Create ml/.venv and install the Python training dependencies

# A real file target, so the ~2 GB dependency install runs once and then only again when
# ml/requirements.txt actually changes. Without the stamp every `make train`, `make
# list-models` and `make convert-model` re-resolves pip first and buries its own output.
$(VENV_STAMP): ml/requirements.txt
	@if [ ! -x "$(PY)" ]; then echo "creating $(VENV)"; $(PYTHON) -m venv $(VENV); fi
	@$(PY) -m pip install --quiet --upgrade pip
	@$(PY) -m pip install --quiet -r ml/requirements.txt
	@mkdir -p $(dir $@)
	@touch $@
	@echo "python environment ready ($(VENV))"

inspect-data: venv ## Show a downloaded dataset's folders and inferred mapping (make inspect-data DIR=~/Downloads/set)
	$(PY) ml/ingest_sources.py --inspect $(DIR)

ingest-data: venv ## Merge downloaded datasets into ml/source (make ingest-data INGEST_ARGS="--source trashnet=~/dl/trashnet")
	$(PY) ml/ingest_sources.py $(INGEST_ARGS)

prepare-data: venv ## Download/organise the training dataset into ml/dataset
	$(PY) ml/prepare_dataset.py $(PREPARE_ARGS)

dataset: venv ## Download RealWaste + TrashNet and rebuild ml/source and ml/dataset for all 10 classes
	$(PY) ml/build_dataset.py $(DATASET_ARGS)

# Linux and WSL2 only: TensorFlow has no CUDA build for native Windows or macOS. The pip
# `and-cuda` extra brings CUDA and cuDNN along, so no system CUDA toolkit is needed - only
# the NVIDIA driver. Pinned to the TensorFlow already in the venv, so nothing else moves.
venv-gpu: venv ## Add NVIDIA CUDA support to ml/.venv (Linux / WSL2; needs the NVIDIA driver)
	@if [ "$$(uname -s)" != "Linux" ]; then \
	   echo "venv-gpu: CUDA TensorFlow needs Linux or WSL2 - see ml/README.md"; exit 1; fi
	$(PY) -m pip install "tensorflow[and-cuda]==$$($(PY) -m pip show tensorflow | sed -n 's/^Version: //p')"
	$(PY) ml/gpu_check.py

gpu-check: venv ## Show whether TensorFlow can see and use an NVIDIA GPU, and why not
	$(PY) ml/gpu_check.py

train-gpu: venv ## Train the recommended EfficientNetV2-B0 recipe on an NVIDIA GPU
	$(PY) ml/gpu_check.py --require --quiet
	$(PY) ml/train.py $(GPU_TRAIN_ARGS) $(GPU_PRECISION) $(TRAIN_ARGS)

train: venv ## Train the custom classifier and export it to models/custom
	$(PY) ml/train.py $(TRAIN_ARGS)

train-smoke: venv ## Prove the whole training + TFJS export toolchain works, with no dataset
	$(PY) ml/train.py $(SMOKE_ARGS) $(TRAIN_ARGS)

evaluate: venv ## Score the trained model on the test split and print a confusion matrix
	$(PY) ml/evaluate.py $(EVAL_ARGS)

list-models: venv ## List the pretrained architectures convert-model can convert
	$(PY) ml/convert_pretrained.py --list

convert-model: venv ## Convert a pretrained ImageNet model to TFJS (make convert-model ARCH=InceptionV3)
	$(PY) ml/convert_pretrained.py --arch $(ARCH) --quantize $(QUANTIZE) $(CONVERT_ARGS)

##@ Maintenance

db-reset: ## Reset the SQLite database (running container, or the host copy) and restart the backend
	@if $(COMPOSE) ps --status running --services 2>/dev/null | grep -qx backend; then \
	   echo "resetting the database inside the running backend container"; \
	   $(COMPOSE) exec -T backend npm run db:reset && $(COMPOSE) restart backend; \
	 else \
	   echo "backend container is not running - removing the host database files"; \
	   rm -f backend/data/ecosort.db backend/data/ecosort.db-wal backend/data/ecosort.db-shm; \
	 fi
	@echo "database reset"

clean: ## Stop the stack, drop volumes, and remove node_modules / build output
	-$(COMPOSE) down -v --remove-orphans
	rm -rf backend/node_modules frontend/node_modules
	rm -rf frontend/dist frontend/.vite
	rm -rf backend/coverage frontend/coverage
	@echo "clean (models/ and ml/.venv were kept - remove them by hand if you want to)"
