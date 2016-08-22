#pragma once

#include "ImageProCom.h"

typedef enum img_segment_type
{
	IMG_SEGMENT_MEANSHIFT,
	IMG_SEGMENT_PYR,
	IMG_SEGMENT_GRABCUT,
	IMG_SEGMENT_ENDGE,
	IMG_SEGMENT_THRESOLD,
	IMG_SEGMENT_WATERSHED,
	IMG_SEGMENT_UNREV
};


class ImageSegment
{
public:
	ImageSegment();
	ImageSegment(int spatialRad,int colorRad,int maxPryLevel):m_spatialRad(spatialRad),m_colorRad(colorRad),m_maxPryLevel(maxPryLevel){}
	ImageSegment(int thresold):m_thresold(thresold){}
	bool imagePro(Mat& src,Mat& dst);
	Mat imageElementFastReplace(Mat& src,int oldValue,int newValue);

	bool imageSegment(Mat& src, Mat& dst,int segmentMode,int spatialRad,int colorRad,int maxPryLevel);
	int  imageElementFastReplaceByOtherImage(Mat& src, Mat& other);
	int  imageElementFastReplace(Mat& src, int oldValue, int newValue);


	~ImageSegment();

private:
	int m_spatialRad;
	int m_colorRad;
	int m_maxPryLevel;
	int m_thresold;

};

