#ifndef IMAGESHAPEFEATURE_H_
#define IMAGESHAPEFEATURE_H_

#include ""

class ImageShapeFeature
{
public:
	ImageShapeFeature();

	void ellipticFourierDescriptor(vector<Point> &contour, vector<float> &CE);





	~ImageShapeFeature();
};

#endif


