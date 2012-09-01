.PHONY : all clean

LIB_OBJECTS=oil-image.o oil-format.o oil-format-png.o oil-palette.o oil-dither.o
CFLAGS=`libpng-config --cflags` -O2
LDFLAGS=`libpng-config --ldflags`

all : test

clean :
	rm -f test main.o
	rm -f ${LIB_OBJECTS}

test : main.o ${LIB_OBJECTS}
	${CC} ${LDFLAGS} main.o ${LIB_OBJECTS} -o test
