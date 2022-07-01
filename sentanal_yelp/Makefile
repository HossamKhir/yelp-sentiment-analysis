dist:	## builds source and wheel package
	python3 setup.py sdist bdist_wheel

clean-build:	## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .egg/

test:	## runs tests quickly with default python
	pytest
