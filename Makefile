PYTHON ?= python
CONFIG ?= configs/default.yaml
RUN = PYTHONPATH=src $(PYTHON) -m linkpred.cli --config $(CONFIG)

.PHONY: help data splits baselines features train evaluate report test all clean-results

help:
	@echo "Targets: data splits baselines features train evaluate report test all"
	@echo "Override config with: make <target> CONFIG=configs/other.yaml"

data:        ## download raw works and build the cleaned co-authorship table
	$(RUN) data

splits:      ## build leak-free temporal train/val/test edges + negatives
	$(RUN) splits

baselines:   ## graph heuristics + Katz under all evaluation protocols -> results/baselines.*
	PYTHONPATH=src $(PYTHON) scripts/run_baselines.py --config $(CONFIG)

features:    ## work embeddings (cached) + pair features per split, pre-cutoff only
	$(RUN) features

train:       ## feature models (LR, LightGBM): mix grid, search, 5 seeds, paired tests, ablation
	PYTHONPATH=src $(PYTHON) scripts/run_feature_models.py --config $(CONFIG)
	PYTHONPATH=src $(PYTHON) scripts/activity_diagnostic.py

evaluate:    ## score models on val/test, write results/*.csv
	$(RUN) evaluate

report:      ## tables and figures generated only from results/*.csv
	$(RUN) report

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

all: data splits baselines features train evaluate report

clean-results:
	rm -f results/*.csv
