# add path
export PYTHONPATH := $(PYTHONPATH):$(abspath $(CURDIR))

# initialization
init:
	pip install -r requirements.txt --no-cache-dir
	git clone https://github.com/matthias-k/DeepGaze models/DeepGaze

# predictions
generate-predictions:
	python src/data/build.py
	python src/features/build.py

evaluate-predictions:
	python src/evaluation/maps.py