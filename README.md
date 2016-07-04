Really, this started with me trying to understand caffe, and to bring it's weights to
a torch implementation. (I also wanted it to be a "pure" nn implementation, and not be
dependent on other libraries (cunn, cudnn, inn, etc.)

But, now, I guess I'm trying to make it so that it can translate more unassisted,
by just giving it a caffemodel, and not having to make a custom torch implementation that
this will just _load_.
