#ifndef INNER_PRODUCT_LAYER_H
#define INNER_PRODUCT_LAYER_H

class inner_product_layer: public layer {
	public:
		inner_product_layer(const std::string& name, const caffe::InnerProductParameter& np) : layer(name, "inner_product_layer"), _inner_productparameter(np) { }
	private:
		const caffe::InnerProductParameter& _inner_productparameter;
};

#endif
