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
using namespace std;

#include "layer.H"
#include "network.H"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

bool ReadProtoFromTextFile(const char* filename, Message &proto) {
	int fd = open(filename, O_RDONLY);
	FileInputStream* input = new FileInputStream(fd);
	bool success = google::protobuf::TextFormat::Parse(input, &proto);
	delete input;
	close(fd);
	return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
	int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	FileOutputStream* output = new FileOutputStream(fd);
	assert(google::protobuf::TextFormat::Print(proto, output));
	delete output;
	close(fd);
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

static void empty(NetParameter& np) {
	for (int i = 0; i < np.layers_size(); ++i) {
		np.mutable_layers(i)->clear_blobs();
	}
}

int main(int argc, char *argv[]) {
	NetParameter np;
//	assert(ReadProtoFromBinaryFile("/usr/local/caffe-2015-07-26/models/bvlc_alexnet/bvlc_alexnet.caffemodel", np));
	assert(ReadProtoFromBinaryFile("/usr/local/caffe-2015-07-26/models/bvlc_googlenet/bvlc_googlenet.caffemodel", np));
//	assert(ReadProtoFromBinaryFile("/usr/local/caffe-2015-07-26/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel", np));
//	assert(ReadProtoFromBinaryFile("/usr/local/caffe-2015-07-26/models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel", np));
//	assert(ReadProtoFromBinaryFile("/usr/local/caffe-2015-07-26/models/finetune_flickr_style/finetune_flickr_style.caffemodel", np));
//	assert(ReadProtoFromTextFile("/usr/local/caffe-2015-07-26/models/bvlc_alexnet/deploy.txt", np));
	NetParameter np2 = np;
	empty(np2);
	WriteProtoToTextFile(np2, "/tmp/wtf.prototxt");
	network n(np);
	n.pass1();
	n.pass2();
	return 0;
}
