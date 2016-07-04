#ifndef LAYER_H
#define LAYER_H

class layer {
	public:
		layer(const std::string& name, const std::string& type) : _name(name), _type(type) { }
		virtual ~layer() { }

		virtual bool top(const std::string& s, layer *l) {
			bool rv = false;
			for (std::vector<std::pair<std::string, layer *> >::iterator i = _tops.begin(); i != _tops.end(); ++i) {
				if ((*i).first == s) {
					rv = true;
					(*i).second = l;
				}
			}
			assert(rv == false);
			_tops.push_back(std::pair<std::string, layer *>(s, l));
			return (true);
		}

		virtual bool accepts_as_consumer(const std::string& s, layer *l) {
			bool rv = false;
			for (std::vector<std::pair<std::string, layer *> >::iterator i = _tops.begin(); i != _tops.end(); ++i) {
				if ((*i).first == s) {
					rv = true;
					(*i).second = l;
				}
			}
			return rv;
		}

		virtual bool bottom(const std::string& s, layer *l) {
			bool rv = false;
			for (std::vector<std::pair<std::string, layer *> >::iterator i = _bottoms.begin(); i != _bottoms.end(); ++i) {
				if (((*i).first == s) && ((*i).second != l)) {
					rv = true;
					(*i).second = l;
				}
			}
			assert(!rv);
			_bottoms.push_back(std::pair<std::string, layer *>(s, l));
			return true;
		}

		const std::string& name() const { return(_name); }
		const std::string& type() const { return(_type); }
	private:
		std::string _name;
		std::string _type;

		std::vector<std::pair<std::string, layer *> > _tops;
		std::vector<std::pair<std::string, layer *> > _bottoms;

		layer(const layer&);
		layer& operator=(const layer&);
};

#endif
