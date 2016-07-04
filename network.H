#ifndef NETWORK_H
#define NETWORK_H

class network {
	public:
		network(const caffe::NetParameter& np) : _netparameter(np) { }
		virtual ~network();
		void pass1();
		void pass2();
	private:
		network& assign(const caffe::V1LayerParameter&, layer *);

		const caffe::NetParameter& _netparameter;
		std::vector<std::pair<std::string, layer *> > _layers;
};

#endif
