#ifndef CONCAT_LAYER_H
#define CONCAT_LAYER_H

class concat_layer: public layer {
	public:
		concat_layer(const std::string& name, const caffe::ConcatParameter& np) : layer(name, "concat_layer"), _concatparameter(np) { }
	private:
		const caffe::ConcatParameter& _concatparameter;
};

#endif
