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

static void dump_blobs(const V1LayerParameter& v1lp) {
	const RepeatedPtrField<BlobProto>& blobs = v1lp.blobs();
	bool first = true;
	for (RepeatedPtrField<BlobProto>::const_iterator i = blobs.begin(); i != blobs.end(); ++i) {
		const BlobProto& blob = (*i);

		if (first) {
			if (v1lp.type() == V1LayerParameter_LayerType_CONVOLUTION) {
				if (v1lp.has_convolution_param()) {
					const ConvolutionParameter& cp = v1lp.convolution_param();
					fprintf(stderr, "	num_output=%d\n", cp.num_output());
					if (!cp.bias_term())
						fprintf(stderr, "	bias_term=%d\n", cp.bias_term());
					fprintf(stderr, "	(kernel=%dx%d)\n", cp.kernel_size(), cp.kernel_size());
					fprintf(stderr, "	group=%d\n", cp.group());
				}
			}
			fprintf(stderr, "	blob has %d elements\n", blob.data_size());
			if (blob.has_shape()) {
				const BlobShape& blobshape = blob.shape();
				bool print_comma = false;
				fprintf(stderr, "	blob has shape ");
				long long int p = 1;
				for (RepeatedField<long long int>::const_iterator j = blobshape.dim().begin(); j != blobshape.dim().end(); ++j) {
					if (print_comma) fprintf(stderr, " x ");
					fprintf(stderr, "%lld", (*j));
					print_comma = true;
					p = p * (*j);
				}
				fprintf(stderr, " = %lld\n", p);
			}
			else {
				long long int p = blob.num() * blob.channels() * blob.height() * blob.width();
				fprintf(stderr, "	blob has no shape but is %d x %d x %d x %d = %lld\n", blob.num(), blob.channels(), blob.height(), blob.width(), p);
			}
		}
		else {
			fprintf(stderr, "	bias is of size %d\n", blob.data_size());
		}
		first = false;
	}
}

int main(int argc, char *argv[]) {
	NetParameter np;
	assert(ReadProtoFromBinaryFile("/usr/local/caffe-2015-07-26/models/bvlc_alexnet/bvlc_alexnet.caffemodel", np));
	fprintf(stderr, "np.layer_size() = %d\n", np.layers_size());
	for (int i = 0; i < np.layers_size(); ++i) {
		const V1LayerParameter& v1lp = np.layers(i);
		if (v1lp.type() == V1LayerParameter_LayerType_CONVOLUTION) {
			fprintf(stderr, "%d: %s\n", i, (v1lp.has_name() ? v1lp.name().c_str() : "(NULL)"));
			dump_blobs(v1lp);
		}
		assert((v1lp.blobs_size() == 0) || (v1lp.blobs_size() == 2));
	}
	return 0;
}
