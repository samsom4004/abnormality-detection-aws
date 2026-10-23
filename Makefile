.PHONY: install test lint demo package docker clean
PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	ruff check src tests mlops scripts

# end-to-end smoke test on a tiny synthetic dataset (NOT medical data; no downloads)
demo:
	abnormality-detection synthetic --out data/synthetic
	abnormality-detection train --data data/synthetic --out artifacts/model --arch vgg16 --image-size 64 --epochs 2 --no-pretrained

package:
	bash scripts/package_source.sh

docker:
	docker build -t abnormality-detection:latest .

clean:
	rm -rf build artifacts data .pytest_cache .ruff_cache
