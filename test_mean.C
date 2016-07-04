#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "caffe.protoc.pb.h"

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::RepeatedPtrField;
using google::protobuf::RepeatedField;
using google::protobuf::Message;
using namespace caffe;

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

bool ReadProtoFromTextFile(const char* filename, Message &proto) {
	int fd = open(filename, O_RDONLY);
	FileInputStream* input = new FileInputStream(fd);
	bool success = google::protobuf::TextFormat::Parse(input, &proto);
	delete input;
	close(fd);
	return success;
}

bool ReadProtoFromBinaryFile(const char* filename, Message &proto) {
	int fd = open(filename, O_RDONLY);
	ZeroCopyInputStream* raw_input = new FileInputStream(fd);
	CodedInputStream* coded_input = new CodedInputStream(raw_input);
	coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

	bool success = proto.ParseFromCodedStream(coded_input);

	delete coded_input;
	delete raw_input;
	close(fd);
	return success;
}

static void fswrite(FILE *fp, const char *str) {
	unsigned int x = 1+strlen(str);
	assert(1 == fwrite(&x, sizeof(int), 1, fp));
	assert(x == fwrite(str, sizeof(char), x, fp));
}

static void flwrite(FILE *fp, BlobProto& bp) {
	int dim = 3;
	long long int channels = bp.channels();
	long long int height = bp.height();
	long long int width = bp.width();
	assert(1 == fwrite(&dim, sizeof(dim), 1, fp));
	assert(1 == fwrite(&channels, sizeof(channels), 1, fp));
	assert(1 == fwrite(&height, sizeof(height), 1, fp));
	assert(1 == fwrite(&width, sizeof(width), 1, fp));
}

static void fdwrite(FILE *fp, unsigned long int sz, const float *data) {
	double *doubledata = new double[sz];
	for (unsigned long int i = 0; i < sz; ++i) doubledata[i] = data[i];

	assert(sz == fwrite(doubledata, sizeof((*doubledata)), sz, fp));

	delete[] doubledata; doubledata = NULL;
}


int main(int argc, char *argv[]) {
	BlobProto bp;
	assert(ReadProtoFromBinaryFile("/usr/local/caffe-2015-07-26/data/ilsvrc12/imagenet_mean.binaryproto", bp));
	fprintf(stderr, "bp has %d elements and is %dx%dx%dx%d\n", bp.data_size(), bp.num(), bp.channels(), bp.height(), bp.width());
	FILE *fp = fopen("/tmp/mean.bin", "w");
	if (fp) {
		fswrite(fp, "nn.DoubleTensor");
		flwrite(fp, bp);
		fdwrite(fp, bp.data_size(), bp.data().data());
		fclose(fp); fp = NULL;
	}
	return 0;
}
