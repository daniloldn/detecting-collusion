.PHONY: preprocess windows feature 

preprocess:
	python src/simulation/run_dgp0.py

windows:
	python src/simulation/windows/run_window.py

feature:
	python src/data/run_feature.py

all: preprocess windows feature