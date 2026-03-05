PYTHON := /c/Users/danil/anaconda3/envs/vdcol/python.exe
.PHONY: preprocess windows feature scoring false

preprocess:
	$(PYTHON) src/simulation/run_dgp0.py

windows:
	$(PYTHON) src/simulation/windows/run_window.py

feature:
	$(PYTHON) src/data/run_feature.py

scoring:
	$(PYTHON) src/scoring/run_scoring.py

false:
	$(PYTHON) src/scoring/fp_calmf.py

all: preprocess windows feature