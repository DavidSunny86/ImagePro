// ImageProMain.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "ImageTrans.h"

int _tmain(int argc, _TCHAR* argv[])
{

	Mat src = imread("E:\\1.jpg");
	ImageTrans it;
	vector<Mat> list = it.getImageColorSplit(src, IMG_COLORSPACE_HSV);


	return 0;
}

