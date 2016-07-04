#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

class convolution_layer: public layer {
	public:
		convolution_layer(const std::string& name, const caffe::ConvolutionParameter& np) : layer(name, "convolution_layer"), _convolutionparameter(np) { }
	private:
		const caffe::ConvolutionParameter& _convolutionparameter;
};

#endif
