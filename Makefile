make: src/main.c src/tests/csvl_test.c src/tests/csvl_filter.c src/libs/csvl/csvl.c src/libs/ocl_wrapper/ocl_wrapper.c src/libs/kernel_launchers/kernel_launchers.c
	gcc -o bin/tests/csvl_test src/tests/csvl_test.c src/libs/csvl/csvl.c
	gcc -o bin/tests/csvl_filter src/tests/csvl_filter.c src/libs/csvl/csvl.c
	gcc -o bin/main src/main.c src/libs/csvl/csvl.c src/libs/ocl_wrapper/ocl_wrapper.c src/libs/kernel_launchers/kernel_launchers.c -framework OpenCL

clean:
	rm bin/tests/csvl_test
	rm bin/tests/csvl_filter
	rm bin/main
