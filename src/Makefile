make: main.c csvl/csvl_test.c csvl/csvl.c
	gcc -o build/csvl_test csvl/csvl_test.c csvl/csvl.c
	gcc -o build/main main.c csvl/csvl.c -framework OpenCL

clean:
	rm build/csvl_test
	rm build/main
