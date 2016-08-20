#pragma once

#include "ImageProCom.h"

typedef enum img_segment_type
{
	IMG_SEGMENT_MEANSHIFT,
	IMG_SEGMENT_GRABCUT,
	IMG_SEGMENT_THRESOLD,
	IMG_SEGMENT_UNREV
};


class ImageSegment
{
public:
	ImageSegment();
	ImageSegment(int spatialRad,int colorRad,int maxPryLevel):m_spatialRad(spatialRad),m_colorRad(colorRad),m_maxPryLevel(maxPryLevel){}
	ImageSegment(int thresold):m_thresold(thresold){}
	bool imagePro(Mat& src,Mat& dst);
	bool imageSegment(Mat& src, Mat& dst,int segmentMode,int spatialRad,int colorRad,int maxPryLevel);


	~ImageSegment();

private:
	int m_spatialRad;
	int m_colorRad;
	int m_maxPryLevel;
	int m_thresold;

};

