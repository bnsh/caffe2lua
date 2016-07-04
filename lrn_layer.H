#ifndef LRN_LAYER_H
#define LRN_LAYER_H

class lrn_layer: public layer {
	public:
		lrn_layer(const std::string& name, const caffe::LRNParameter& np) : layer(name, "lrn_layer"), _lrnparameter(np) { }
	private:
		const caffe::LRNParameter& _lrnparameter;
};

#endif
