# add path
export PYTHONPATH := $(PYTHONPATH):$(abspath $(CURDIR))

# initialization
init:
	pip install -r requirements.txt --no-cache-dir

# 
generate-predictions:
	python src/data/build_data.py
	python src/features/build_maps.py

evaluate-predictions:
	python src/features/eval_maps.py