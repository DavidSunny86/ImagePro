#ifndef  IMAGEBASEUTILS_H_
#define  IMAGEBASEUTILS_H_

#define IMAGEBASEUTILS_MODEL_EXPORT __declspec(dllimport)

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


#define CTPLAR_ERR_OUT(fmt,...)   printf("%s(%d)-<%s>: "##fmt, __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)


typedef enum IMAGE_BASE_ERROR_CODE
{
	CTPLAR_OK,
	CTPLAR_INPUT_ERR,
	CTPLAR_LOGIC_ERR,
	CTPLAR_UNREV
};




#endif


