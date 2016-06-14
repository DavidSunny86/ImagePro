// ImageProMain.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "ImageTrans.h"

int _tmain(int argc, _TCHAR* argv[])
{

	Mat src = imread("E:\\1.jpg");
	ImageTrans it;
	vector<Mat> list = it.getImageColorSplit(src, IMG_COLORSPACE_HSV);

	imwrite("H.png",list[0]);
	imwrite("S.png", list[1]);
	imwrite("V.png", list[2]);



	return 0;
}

