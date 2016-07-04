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
using google::protobuf::Message;
using std::string;
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

static void flwrite(FILE *fp, unsigned long int sz) {
	int dim = 1;
	assert(1 == fwrite(&dim, sizeof(dim), 1, fp));
	assert(1 == fwrite(&sz, sizeof(sz), 1, fp));
}

static void fdwrite(FILE *fp, unsigned long int sz, const float *data) {
	double *doubledata = new double[sz];
	for (unsigned long int i = 0; i < sz; ++i) doubledata[i] = data[i];

	assert(sz == fwrite(doubledata, sizeof((*doubledata)), sz, fp));

	delete[] doubledata; doubledata = NULL;
}

static void dump_convolution(int layernum, const string& name, const V1LayerParameter& layer) {
	assert(layer.blobs_size() == 2);
	const BlobProto& weights = layer.blobs(0);
	const BlobProto& bias = layer.blobs(1);
	const ConvolutionParameter& cp = layer.convolution_param();
	int groups = cp.group();
/*
 * a 0 + b == 0
 * a 1 + b == sz/2
 * a = sz/2
 */
	for (int j = 0; j < groups; ++j) {
		char fn[1024];
		if (groups <= 1) snprintf(fn, 1024, "/tmp/bnet/layer-%d-%d.bin", layernum, (1+j));
		else snprintf(fn, 1024, "/tmp/bnet/layer-%d-2-%d.bin", layernum, (1+j));
		fprintf(stderr, "%s\n", fn);
		FILE *fp = fopen(fn, "w");
		if (fp) {
			// first write the name.
			fswrite(fp, "nn.SpatialConvolution");
			flwrite(fp, (weights.data_size() + bias.data_size()) / groups);
			int idx = j;
			unsigned long int woffset = idx * weights.data_size() / groups;
fprintf(stderr, "%s: %d -> %d woffset=%lu\n", fn, j, idx, woffset);
			fdwrite(fp, weights.data_size() / groups, weights.data().data() + woffset);
			unsigned long int boffset = idx * bias.data_size() / groups;
			fdwrite(fp, bias.data_size() / groups, bias.data().data() + boffset);
			
			fclose(fp); fp = NULL;
		}
	}
}

static void dump_linear(int i, const string& name, const V1LayerParameter& layer) {
	assert(layer.blobs_size() == 2);
	const BlobProto& weights = layer.blobs(0);
	const BlobProto& bias = layer.blobs(1);
	char fn[1024];
	if (i == 6) snprintf(fn, 1024, "/tmp/bnet/layer-%d-2.bin", i);
	else snprintf(fn, 1024, "/tmp/bnet/layer-%d-1.bin", i);
	fprintf(stderr, "%s\n", fn);
	FILE *fp = fopen(fn, "w");
	if (fp) {
		// first write the name.
		fswrite(fp, "nn.Linear");
		flwrite(fp, weights.data_size() + bias.data_size());
		fdwrite(fp, weights.data_size(), weights.data().data());
		fdwrite(fp, bias.data_size(), bias.data().data());

		fclose(fp); fp = NULL;
	}
}

int main(int argc, char *argv[]) {
	NetParameter np;
	assert(ReadProtoFromBinaryFile("/usr/local/caffe-2015-07-26/models/bvlc_alexnet/bvlc_alexnet.caffemodel", np));
	int idx = 0;
	for (int i = 0; i < np.layers_size(); ++i) {
		const V1LayerParameter& v1lp = np.layers(i);
/*
0: data [groups=1]
1: conv1 [(3->96 11x11)(1->1 96x1)groups=1]
2: relu1 [groups=1]
3: norm1 [groups=1]
4: pool1 [groups=1]
5: conv2 [(48->256 5x5)(1->1 256x1)groups=2]
6: relu2 [groups=1]
7: norm2 [groups=1]
8: pool2 [groups=1]
9: conv3 [(256->384 3x3)(1->1 384x1)groups=1]
10: relu3 [groups=1]
11: conv4 [(192->384 3x3)(1->1 384x1)groups=2]
12: relu4 [groups=1]
13: conv5 [(192->256 3x3)(1->1 256x1)groups=2]
14: relu5 [groups=1]
15: pool5 [groups=1]
16: fc6 [(1->1 9216x4096)(1->1 4096x1)groups=1]
17: relu6 [groups=1]
18: drop6 [groups=1]
19: fc7 [(1->1 4096x4096)(1->1 4096x1)groups=1]
20: relu7 [groups=1]
21: drop7 [groups=1]
22: fc8 [(1->1 4096x1000)(1->1 1000x1)groups=1]
23: loss [groups=1]
 */
		// So, we need to dump only convs and fc's
		const string& name = v1lp.name();
		V1LayerParameter_LayerType type = v1lp.type();

		if (type == V1LayerParameter_LayerType_CONVOLUTION) dump_convolution(++idx, name, v1lp);
		else if (type == V1LayerParameter_LayerType_INNER_PRODUCT) dump_linear(++idx, name, v1lp);
	}
	return 0;
}
