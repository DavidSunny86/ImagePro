#ifndef  IMAGEBASEUTILS_H_
#define  IMAGEBASEUTILS_H_

#define IMAGEBASEUTILS_MODEL_EXPORT __declspec(dllimport)

#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>
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


#define CT_ONE    (1)
#define CT_TWO    (2)
#define CT_THREE  (3)
#define CT_FOUR   (4) 
#define CT_FIVE   (5) 
#define CT_SIX    (6) 
#define CT_SEVEN  (7)
#define CT_EIGHT  (8) 
#define CT_NINE   (9) 
#define CT_TEN    (10)
#define CT_ZERO   (0)


const string getCurrentSystemTime();

long LastWishesCrashHandler(EXCEPTION_POINTERS *pException);












#endif


