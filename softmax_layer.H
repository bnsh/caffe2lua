#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

class softmax_layer: public layer {
	public:
		softmax_layer(const std::string& name, const caffe::SoftmaxParameter& np) : layer(name, "softmax_layer"), _softmax_parameter(np) { }
	private:
		const caffe::SoftmaxParameter& _softmax_parameter;
};

#endif
