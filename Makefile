CC=g++
CFLAGS=-g3 -O2 -Wall -Werror
LIBS=-L/usr/local/lib -lprotobuf -lopencv_core -lopencv_highgui

# https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto
PROTOS=\
	caffe.protoc \

PROTOS_DIRS=$(PROTOS:.protoc=)
PROTOS_CSOURCES=$(PROTOS:.protoc=.protoc.pb.C)
PROTOS_HEADERS=$(PROTOS:.protoc=.protoc.pb.h)
PROTOS_PYTHONS=$(PROTOS:.protoc=_pb2.py)
PROTOS_PYTHONCOMPILED=$(PROTOS:.protoc=_pb2.pyc)
PROTOS_GENERATED=$(PROTOS_CSOURCES) $(PROTOS_HEADERS) $(PROTOS_PYTHONS)
PROTOS_OBJS=$(PROTOS_CSOURCES:C=o)

SRCS=\
	caffe2lua.C \
	dump_alexnet.C \
	imread.C \
	network.C \
	test.C \
	test_mean.C \

CANDIDATES=$(filter %.d,$(patsubst %.C,%.d,$(patsubst %.c,%.d,$(SRCS))))
DEPS=$(join $(dir $(CANDIDATES)),$(addprefix .,$(notdir $(CANDIDATES))))

OBJS=$(SRCS:C=o) $(PROTOS_OBJS)

BINS=\
	caffe2lua \
	dump_alexnet \
	imread \
	test \
	test_mean \

all: $(PROTOS_GENERATED) $(OBJS) $(BINS)

clean:
	/bin/rm -f $(OBJS) $(PROTOS_GENERATED) $(BINS) $(PROTOS_PYTHONCOMPILED)
	/bin/rm -fr $(PROTOS_DIRS)

checkin:
	/usr/bin/ci -l -m- -t- Makefile $(PROTOS) $(SRCS)

caffe2lua: caffe2lua.o network.o caffe.protoc.pb.o
	$(CC) $(CFLAGS) $(^) -o $(@) $(LIBS)

caffe2lua.o: caffe2lua.C
	$(CC) -c $(CFLAGS) $(filter %.C, $(^)) -o $(@)

dump_alexnet: dump_alexnet.o caffe.protoc.pb.o
	$(CC) $(CFLAGS) $(^) -o $(@) $(LIBS)

dump_alexnet.o: dump_alexnet.C
	$(CC) -c $(CFLAGS) $(filter %.C, $(^)) -o $(@)

imread: imread.o caffe.protoc.pb.o
	$(CC) $(CFLAGS) $(^) -o $(@) $(LIBS)

imread.o: imread.C
	$(CC) -c $(CFLAGS) $(filter %.C, $(^)) -o $(@)

test: test.o caffe.protoc.pb.o
	$(CC) $(CFLAGS) $(^) -o $(@) $(LIBS)

test.o: test.C
	$(CC) -c $(CFLAGS) $(filter %.C, $(^)) -o $(@)

test_mean: test_mean.o caffe.protoc.pb.o
	$(CC) $(CFLAGS) $(^) -o $(@) $(LIBS)

test_mean.o: test_mean.C
	$(CC) -c $(CFLAGS) $(filter %.C, $(^)) -o $(@)

%.protoc.pb.C %.protoc.pb.h %_pb2.py: %.protoc
	/usr/local/bin/protoc --cpp_out=./ --python_out=./ $(*).protoc && \
	mv $(*).protoc.pb.cc $(*).protoc.pb.C && \
	mv $(*)/protoc_pb2.py $(*)_pb2.py && \
	rm -fr $(*)

%.o: %.C
	$(CC) -c $(CFLAGS) $(filter %.C,$(^)) -o $(@)

.%.d: %.C
	@$(CC) $(CFLAGS) -MT $(patsubst %.C,%.o,$(patsubst %.c,%.o,$(<))) -M $(<) -o $(@)

-include $(DEPS)
