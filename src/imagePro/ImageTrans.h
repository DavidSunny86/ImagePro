#pragma once
#include "ImageProCom.h"
#include "cv.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace  cv;

typedef enum COLOR_SPACE
{
	IMG_COLORSPACE_HSV,
	IMG_COLORSAPCE_LAB,
	IMG_COLORSPACE_HSI,
	IMG_COLORSPACE_BGR,
	IMG_COLORSPACE_BGRA,
	IMG_COLORSPACE_HLS,
	IMG_COLORSPACE_UNSER
};

class IMAGEPRO_MODEL_EXPORT ImageTrans
{
public:
	ImageTrans();
	vector<Mat> getImageColorSplit(Mat& srcMat, int mode/*颜色分离颜色空间*/);
	Mat getTargetImage(Mat& srcmat, int minValue, int maxValue);

	~ImageTrans();
};

