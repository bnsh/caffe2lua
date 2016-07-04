#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

class dropout_layer: public layer {
	public:
		dropout_layer(const std::string& name, const caffe::DropoutParameter& np) : layer(name, "dropout_layer"), _dropoutparameter(np) { }
	private:
		const caffe::DropoutParameter& _dropoutparameter;
};

#endif
