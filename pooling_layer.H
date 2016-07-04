#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

class pooling_layer: public layer {
	public:
		pooling_layer(const std::string& name, const caffe::PoolingParameter& np) : layer(name, "pooling_layer"), _poolingparameter(np) { }
	private:
		const caffe::PoolingParameter& _poolingparameter;
};

#endif
