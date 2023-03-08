.PHONY: build dist redist install install-from-source clean uninstall

build:
	python ./setup.py build

dist:
	python ./setup.py sdist bdist_wheel

redist: clean dist

install:
	pip install . --user

install-from-source: dist
	pip install dist/c_unipolator

clean:
	$(RM) -r build dist src/*.egg-info
	$(RM) -r src/unipolator/{unipolator.c} 
	$(RM) -r .pytest_cache
	find . -name __pycache__ -exec rm -r {} +
	#git clean -fdX

uninstall:
	pip uninstall unipolator
