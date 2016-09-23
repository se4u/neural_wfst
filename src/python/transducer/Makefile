all: clean
	python setup.py build_ext; \
	cp build/lib.*-x86_64-2.7/python/transducer/src/transducer.so src/ ;

clean:
	echo "Running $@"; python setup.py clean; rm -rf build/
