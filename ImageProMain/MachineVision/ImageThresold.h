#ifndef IMAGETHRESOLD_H_
#define IMAGETHRESOLD_H_


#include "ImageThrUtils.h"

typedef enum Image_THRESOLD_TYPE
{
	IMAGE_THRESOLD_GLOBAL,
	IMAGE_THRESOLD_LOCAL,
	IMAGE_THRESOLD_OSTU,
	IMAGE_THRESOLD_ITERATION,
	IMAGE_THRESOLD_ADAPTIVE,
	IMAGE_THRESOLD_TOPDOWN,
	IMAGE_THRESOLD_MAXENTROPY,
	Image_thresold_bersen,
	image_thresold_
	IMAGE_THRESOLD_UNREV
};

class ImageThresold
{

public:
	ImageThresold();

	static int getThresoldImage(const Mat& src, Mat& thrMat, int mode);
	static Mat globalThresold(const Mat& src, int params);
	static Mat localThresold(const Mat& src);
	static int ostuThresold(const Mat& src, Mat& dst);
	static Mat maxEntropy(const Mat& src, int mode);
	~ImageThresold();

private:
	int         ostuThresold(const Mat& src, Mat& dst);
	double   caculateCurrentEntropy(CvHistogram * Hist, int curThr, EntropyState state);
	int         calculateBestGlobalThr(const Mat& src);


private:
	Mat m_src;
	Mat m_binary;


private:
	typedef enum { back, object } EntropyState;
	int HistBins = 256;
	float HistRange1[2] = { 0, 255 };
	float *HistRange[1] = { &HistRange1[0] };

};


#endif



