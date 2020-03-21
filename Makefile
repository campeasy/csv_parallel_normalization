make: src/main.c src/tests/csvl_test.c src/tests/csvl_filter.c src/libs/csvl/csvl.c
	gcc -o bin/tests/csvl_test src/tests/csvl_test.c src/libs/csvl/csvl.c
	gcc -o bin/tests/csvl_filter src/tests/csvl_filter.c src/libs/csvl/csvl.c
	gcc -o bin/main src/main.c src/libs/csvl/csvl.c -framework OpenCL

clean:
	rm bin/tests/csvl_test
	rm bin/tests/csvl_filter
	rm bin/main
