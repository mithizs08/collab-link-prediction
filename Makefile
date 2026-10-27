PYTHON ?= python
CONFIG ?= configs/default.yaml
RUN = PYTHONPATH=src $(PYTHON) -m linkpred.cli --config $(CONFIG)

.PHONY: help data splits baselines features train evaluate report test all clean-results gnn-bundle gnn-dry gnn-import gnn-analysis phase7 phase8 final-pass final

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
	PYTHONPATH=src $(PYTHON) scripts/run_node2vec.py --config $(CONFIG)
	PYTHONPATH=src $(PYTHON) scripts/node2vec_quality.py

gnn-bundle:  ## node features + colab_bundle.tar.gz for the Colab GPU run (not committed)
	PYTHONPATH=src $(PYTHON) -c "from linkpred.config import load_config; from linkpred.features.node_features import build_node_features; build_node_features(load_config('$(CONFIG)'))"
	PYTHONPATH=src $(PYTHON) scripts/make_colab_bundle.py

gnn-dry:     ## local CPU dry run of the whole GNN pipeline (2 epochs, 1 seed) into /tmp/gnn_dry
	PYTHONPATH=src $(PYTHON) scripts/run_gnn.py --out /tmp/gnn_dry --device cpu --dry

gnn-import:  ## make gnn-import SRC=~/Downloads/gnn_results
	PYTHONPATH=src $(PYTHON) scripts/import_colab_results.py $(SRC)

gnn-analysis: ## re-evaluate GNN scores, paired tests, hybrid, shortcut check, results/gnn.md (after gnn-import)
	PYTHONPATH=src $(PYTHON) scripts/gnn_analysis.py

phase7:      ## robustness (validation then test): grid edge, two-stage model, disambiguation sensitivity
	PYTHONPATH=src $(PYTHON) scripts/grid_edge.py --config $(CONFIG)
	PYTHONPATH=src $(PYTHON) scripts/two_stage.py --config $(CONFIG) --splits val test
	PYTHONPATH=src $(PYTHON) scripts/disambiguation_sensitivity.py --config $(CONFIG) --splits val test
	PYTHONPATH=src $(PYTHON) scripts/rq2_text_ablation.py --config $(CONFIG) --splits val test

phase8:      ## explainability + error analysis (validation first, then test) -> docs/explainability.md
	PYTHONPATH=src $(PYTHON) scripts/phase8_shap.py --config $(CONFIG) --split val
	PYTHONPATH=src $(PYTHON) scripts/phase8_shap.py --config $(CONFIG) --split test
	PYTHONPATH=src $(PYTHON) scripts/rq2_text_ablation.py --config $(CONFIG) --block-order --splits val test
	PYTHONPATH=src $(PYTHON) scripts/phase8_gnn_explain.py --config $(CONFIG)
	PYTHONPATH=src $(PYTHON) scripts/extract_work_fields.py --config $(CONFIG)
	PYTHONPATH=src $(PYTHON) scripts/phase8_errors.py --config $(CONFIG) --split val
	PYTHONPATH=src $(PYTHON) scripts/phase8_errors.py --config $(CONFIG) --split test
	PYTHONPATH=src $(PYTHON) scripts/phase8_calibration.py --config $(CONFIG) --splits val test
	PYTHONPATH=src $(PYTHON) scripts/final_comparison.py --holm-only
	PYTHONPATH=src $(PYTHON) scripts/phase8_report.py

final-pass:  ## reviewer checks (exploratory): pair-focused LR, validation-best gains, node2vec sanity, clustered bootstrap
	PYTHONPATH=src $(PYTHON) scripts/final_pass_checks.py --config $(CONFIG) --splits val test

final:       ## primary comparison (34-test Holm family incl. phase7 tests, 5,000 resamples) + results/final_summary.md
	PYTHONPATH=src $(PYTHON) scripts/final_comparison.py
	PYTHONPATH=src $(PYTHON) scripts/final_summary.py

evaluate:    ## score models on val/test, write results/*.csv
	$(RUN) evaluate

report:      ## tables and figures generated only from results/*.csv
	$(RUN) report

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

all: data splits baselines features train evaluate report

clean-results:
	rm -f results/*.csv
