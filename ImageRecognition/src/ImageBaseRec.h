#pragma once

#include "ImageRecUtils.h"

class ImageBaseRec
{
public:
	ImageBaseRec();

	virtual Mat getTargetRoi(const Mat& src, int type);
	virtual Mat rotateTargetImg(const Mat& src);
	virtual bool splitImg(const Mat& src, vector<Mat>& splitImg, int splitMode);


	virtual ~ImageBaseRec();
};
