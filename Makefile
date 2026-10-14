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

baselines:   ## graph heuristics (CN, Jaccard, Adamic-Adar, PA, ...)
	$(RUN) baselines

features:    ## pair features for LightGBM, computed per-split from pre-cutoff graph only
	$(RUN) features

train:       ## LightGBM, node2vec, GraphSAGE, GAT over all configured seeds
	$(RUN) train

evaluate:    ## score models on val/test, write results/*.csv
	$(RUN) evaluate

report:      ## tables and figures generated only from results/*.csv
	$(RUN) report

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

all: data splits baselines features train evaluate report

clean-results:
	rm -f results/*.csv
