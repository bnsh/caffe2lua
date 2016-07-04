#include <string>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "caffe.protoc.pb.h"
#include "layer.H"
#include "network.H"
#include "convolution_layer.H"
#include "data_layer.H"
#include "image_data_layer.H"
#include "relu_layer.H"
#include "lrn_layer.H"
#include "pooling_layer.H"
#include "inner_product_layer.H"
#include "dropout_layer.H"
#include "softmax_layer.H"
#include "split_layer.H"
#include "concat_layer.H"

using namespace caffe;
using namespace std;

void network::pass1() {
// Pass 1 simply accumulates all the nodes.
// Pass 2 will make all the links.
	int numlayers = _netparameter.layers_size();
	for (int i = 0; i < numlayers; ++i) {
		const V1LayerParameter& v1lp = _netparameter.layers(i);
		assert(v1lp.has_name());
		switch (v1lp.type()) {
			case V1LayerParameter_LayerType_CONVOLUTION: {
				assign(v1lp, new convolution_layer(v1lp.name(), v1lp.convolution_param()));
				break;
			}
			case V1LayerParameter_LayerType_DATA: {
// TODO: Leaving off here. Insight: data takes two "top" parameters (only?) and the first is the
// input and the second is the target.
				assign(v1lp, new data_layer(v1lp.name(), v1lp.data_param()));
				break;
			}
			case V1LayerParameter_LayerType_DROPOUT: {
				assign(v1lp, new dropout_layer(v1lp.name(), v1lp.dropout_param()));
				break;
			}
			case V1LayerParameter_LayerType_INNER_PRODUCT: {
				assign(v1lp, new inner_product_layer(v1lp.name(), v1lp.inner_product_param()));
				break;
			}
			case V1LayerParameter_LayerType_LRN: {
				assign(v1lp, new lrn_layer(v1lp.name(), v1lp.lrn_param()));
				break;
			}
			case V1LayerParameter_LayerType_POOLING: {
				assign(v1lp, new pooling_layer(v1lp.name(), v1lp.pooling_param()));
				break;
			}
			case V1LayerParameter_LayerType_RELU: {
				assign(v1lp, new relu_layer(v1lp.name(), v1lp.relu_param()));
				break;
			}
			case V1LayerParameter_LayerType_SOFTMAX_LOSS: {
				assign(v1lp, new softmax_layer(v1lp.name(), v1lp.softmax_param()));
				break;
			}
			case V1LayerParameter_LayerType_SPLIT: {
				assign(v1lp, new split_layer(v1lp.name()));
				break;
			}
			case V1LayerParameter_LayerType_CONCAT: {
				assign(v1lp, new concat_layer(v1lp.name(), v1lp.concat_param()));
				break;
			}
			case V1LayerParameter_LayerType_IMAGE_DATA: {
				assign(v1lp, new image_data_layer(v1lp.name(), v1lp.image_data_param()));
				break;
			}
			case V1LayerParameter_LayerType_ABSVAL: fprintf(stderr, "V1LayerParameter_LayerType_ABSVAL is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_ACCURACY: fprintf(stderr, "V1LayerParameter_LayerType_ACCURACY is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_ARGMAX: fprintf(stderr, "V1LayerParameter_LayerType_ARGMAX is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_BNLL: fprintf(stderr, "V1LayerParameter_LayerType_BNLL is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_CONTRASTIVE_LOSS: fprintf(stderr, "V1LayerParameter_LayerType_CONTRASTIVE_LOSS is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_DECONVOLUTION: fprintf(stderr, "V1LayerParameter_LayerType_DECONVOLUTION is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_DUMMY_DATA: fprintf(stderr, "V1LayerParameter_LayerType_DUMMY_DATA is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_EUCLIDEAN_LOSS: fprintf(stderr, "V1LayerParameter_LayerType_EUCLIDEAN_LOSS is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_ELTWISE: fprintf(stderr, "V1LayerParameter_LayerType_ELTWISE is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_EXP: fprintf(stderr, "V1LayerParameter_LayerType_EXP is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_FLATTEN: fprintf(stderr, "V1LayerParameter_LayerType_FLATTEN is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_HDF5_DATA: fprintf(stderr, "V1LayerParameter_LayerType_HDF5_DATA is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_HDF5_OUTPUT: fprintf(stderr, "V1LayerParameter_LayerType_HDF5_OUTPUT is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_HINGE_LOSS: fprintf(stderr, "V1LayerParameter_LayerType_HINGE_LOSS is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_IM2COL: fprintf(stderr, "V1LayerParameter_LayerType_IM2COL is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_INFOGAIN_LOSS: fprintf(stderr, "V1LayerParameter_LayerType_INFOGAIN_LOSS is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_MEMORY_DATA: fprintf(stderr, "V1LayerParameter_LayerType_MEMORY_DATA is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS: fprintf(stderr, "V1LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_MVN: fprintf(stderr, "V1LayerParameter_LayerType_MVN is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_POWER: fprintf(stderr, "V1LayerParameter_LayerType_POWER is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_SIGMOID: fprintf(stderr, "V1LayerParameter_LayerType_SIGMOID is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_SIGMOID_CROSS_ENTROPY_LOSS: fprintf(stderr, "V1LayerParameter_LayerType_SIGMOID_CROSS_ENTROPY_LOSS is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_SILENCE: fprintf(stderr, "V1LayerParameter_LayerType_SILENCE is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_SOFTMAX: fprintf(stderr, "V1LayerParameter_LayerType_SOFTMAX is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_SLICE: fprintf(stderr, "V1LayerParameter_LayerType_SLICE is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_TANH: fprintf(stderr, "V1LayerParameter_LayerType_TANH is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_WINDOW_DATA: fprintf(stderr, "V1LayerParameter_LayerType_WINDOW_DATA is unimplemented!\n"); assert(false); break;
			case V1LayerParameter_LayerType_THRESHOLD: fprintf(stderr, "V1LayerParameter_LayerType_THRESHOLD is unimplemented!\n"); assert(false); break;
			default:
				fprintf(stderr, "Unknown V1LayerParameterType %d\n", v1lp.type());
				assert(false);
				break;
		}

	}
}

void network::pass2() {
// Pass 1 simply accumulates all the nodes.
// Pass 2 will make all the links.
// So, now we rip through each layer and look for bottoms.
// I suppose top is simply a "group"? NO. top is effectively
// "output". So, SPLIT for instance can have many "tops".
// Also. tops are not unique. *sigh*. They seem to be "sequential", kind of.
// See "conv1/7x7_s2" and "conv1/relu_7x7"..
//	Both claim to produce "conv1/7x7_s2" as their output.
	unsigned int numlayers = _netparameter.layers_size();
	assert(numlayers == _layers.size());
	for (unsigned int i = 0; i < numlayers; ++i) {
		const V1LayerParameter& v1lp = _netparameter.layers(i);
		assert(v1lp.has_name());
		int bottoms = v1lp.bottom_size();
		layer *consumer = _layers[i].second;
		for (int j = 0; j < bottoms; ++j) {
			const string& producer_name = v1lp.bottom(j);
			for (unsigned int k = i-1; k >= 0; k--) {
				layer *producer = _layers[k].second;
// We need to find the node that produces something called producer_name
				if (producer->accepts_as_consumer(producer_name, consumer)) {
					assert(consumer->bottom(producer_name, producer));
					break;
				}
			}
		}
	}
}

network& network::assign(const V1LayerParameter& v1lp, layer *l) {
	_layers.push_back(pair<string, layer *>(v1lp.name(),l));

	for (int i = 0; i < v1lp.top_size(); ++i) l->top(v1lp.top(i), NULL);
	return (*this);
}

network::~network() {
	for (vector<pair<string, layer *> >::iterator i = _layers.begin(); i != _layers.end(); ++i) {
		delete (*i).second;
		(*i).second = NULL;
	}
	_layers.erase(_layers.begin(), _layers.end());
}
