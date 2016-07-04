#ifndef IMAGE_DATA_LAYER_H
#define IMAGE_DATA_LAYER_H

class image_data_layer: public layer {
	public:
		image_data_layer(const std::string& name, const caffe::ImageDataParameter& np) : layer(name, "image_data_layer"), _image_dataparameter(np) { }
	private:
		const caffe::ImageDataParameter& _image_dataparameter;
};

#endif
