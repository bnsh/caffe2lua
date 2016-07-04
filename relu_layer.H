#ifndef RELU_LAYER_H
#define RELU_LAYER_H

class relu_layer: public layer {
	public:
		relu_layer(const std::string& name, const caffe::ReLUParameter& np) : layer(name, "relu_layer"), _reluparameter(np) { }
	private:
		const caffe::ReLUParameter& _reluparameter;
};

#endif
