// ImageProMain.cpp : �������̨Ӧ�ó������ڵ㡣
#include "../imagepro/ImageTrans.h"

int main()
{

	Mat src = imread("E:\\1.jpg");
	ImageTrans it;
	vector<Mat> list = it.getImageColorSplit(src, IMG_COLORSPACE_HSV);

	imwrite("H.png",list[0]);
	imwrite("S.png", list[1]);
	imwrite("V.png", list[2]);
	return 0;
}

