#ifndef DATA_LAYER_H
#define DATA_LAYER_H

class data_layer: public layer {
	public:
		data_layer(const std::string& name, const caffe::DataParameter& np) : layer(name, "data_layer"), _dataparameter(np) { }
	private:
		const caffe::DataParameter& _dataparameter;
};

#endif
