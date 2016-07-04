#ifndef IMAGEDEFDECUTILS_H_
#define IMAGEDEFDECUTILS_H_

#define IMAGEDEFDET_MODEL_EXPORT __declspec(dllimport)

#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

typedef enum IMAGEDEFDEC_MATERIALTYPE
{
	IMAGE_METERIAL_PCB,
	IMAGE_METERIAL_PAPER,
	IMAGE_METERIAL_UNREV

};



#endif
