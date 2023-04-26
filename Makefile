.PHONY: build dist redist install install-from-source clean uninstall

build:
	python ./setup.py build

dist:
	python ./setup.py sdist bdist_wheel

redist: clean dist

install:
	CYTHONIZE=1  pip install .

install-from-source: dist
	pip install dist\c_unipolator

clean:
	rd /s /q build
	rd /s /q dist 
	rd /s /q src\unipolator.egg-info
	del  src\unipolator\*.c

gitclean:
	git clean -fdX

uninstall:
	pip uninstall unipolator
