CC        = gcc

LIBS      = -lm -lnetcdf

CFLAGS    = -O -c

OBJ = orbitinterp.o
INC = orbitinterp.h

orbitinterp: $(OBJ)
	$(CC) $(OBJ) $(LIBS) -o orbitinterp.e

orbitinterp.o: orbitinterp.c $(INC)
	$(CC) $(CFLAGS) orbitinterp.c

clean :
	/bin/rm -f *.o orbitinterp.e a.out *~

