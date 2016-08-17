#pragma once

#include "ImageRecUtils.h"

class ImageBaseRec
{
public:
	ImageBaseRec();

	virtual Mat getTargetRoi(const Mat& src, int type);
	virtual Mat rotateTargetImg(const Mat& src);

	virtual 

	virtual ~ImageBaseRec();
};

