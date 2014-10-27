MVE_ROOT := ../../../mve
PSS_ROOT := ../..
TARGET := $(shell basename `pwd`)
include ${MVE_ROOT}/Makefile.inc

CXXFLAGS += -I${MVE_ROOT}/libs -I${PSS_ROOT}/libs ${OPENMP}
LDLIBS += -lpng -ltiff -ljpeg ${OPENMP}

vpath libmvs.a ${PSS_ROOT}/libs/mvs/
SOURCES := $(wildcard [^_]*.cc)
${TARGET}: ${SOURCES:.cc=.o} libmvs.a libmve.a libmve_util.a
	$(LINK.o) $^ $(LDLIBS) -o $@ ${OPENMP}

clean:
	${RM} ${TARGET} *.o Makefile.dep

.PHONY: clean
