#ifndef IMAGESHAPEFEATURE_H_
#define IMAGESHAPEFEATURE_H_

#include "ImageBaseUtils.h"

typedef enum SHAPETYPE
{
	IMG_SHAPE_HU,
	IMG_SHAPE_SHAPECONTENT,
	IMG_SHAPE_UNREV,
};

class ImageShapeFeature
{
public:
	ImageShapeFeature();

	void ellipticFourierDescriptor(vector<Point> &contour, vector<float> &CE);

	vector<Point> shapeContextFeature(const Mat& currentQuery, int n = 300); // shapeContext ÌØÕ÷
	float computerShapeContext(Mat& src,Mat& dst,int n = 300,int width = 300,int height = 300);
	float computerShapeContext(vector<Point>& src, vector<Point>& dst);

	~ImageShapeFeature();
};

#endif


