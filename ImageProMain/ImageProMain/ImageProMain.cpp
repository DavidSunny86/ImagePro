// ImageProMain.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "../../ImageBeauty/src/ImageBeauty.h"


/*
*	Author: John Hany
*	Website: http://johnhany.net
*	Source code updates: https://github/johnhany/textRotCorrect
*	If you have any advice, you could contact me at: johnhany@163.com
*	Need OpenCV environment!
*
*/

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define GRAY_THRESH 150
#define HOUGH_VOTE 100

//#define DEGREE 27

//int _tmain(int argc, _TCHAR* argv[])
//{
//	ImageSkin is;
//	Mat srcImg = imread("D:/1.png");
//
//
//	Mat dst = is.skinWhiteing(srcImg);
//
//	return 0;
//}

