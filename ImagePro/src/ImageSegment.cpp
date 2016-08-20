#include "ImageSegment.h"

ImageSegment::ImageSegment()
{}

bool ImageSegment::imageSegment(Mat& src, Mat& dst, int segmentMode, int spatialRad,int colorRad, int maxPryLevel)
{
	if (src.empty())
	{
		return false;
	}

	switch (segmentMode)
	{
	case 0:
	{
		pyrMeanShiftFiltering(src, dst, spatialRad, colorRad, maxPryLevel);
		RNG rng = theRNG();
		Mat mask(dst.rows + 2, dst.cols + 2, CV_8UC1, Scalar::all(0));
		for (int i = 0; i < dst.rows; i++)
		{
			for (int j = 0; j < dst.cols; j++)
			{
				if (mask.at<uchar>(i + 1, j + 1) == 0)
				{
					Scalar newcolor(rng(256), rng(256), rng(256));
					floodFill(dst, mask, Point(j, i), newcolor, 0, Scalar::all(1), Scalar::all(1));
				}
			}
		}
	}
	break;
	default:
		break;
	}

	return  true;
}


ImageSegment::~ImageSegment()
{}



int main(int argc, uchar* argv[])
{

	namedWindow("src", WINDOW_AUTOSIZE);
	namedWindow("dst", WINDOW_AUTOSIZE);


	Mat src, dst;

	src = imread("E:\\1.png");
	CV_Assert(!src.empty());

	int spatialRad = 10;
	int colorRad = 10;
	int maxPryLevel = 1;

	//虽然createTrackbar函数的参数onChange函数要求其2个参数形式为onChange(int,void*)
	//但是这里是系统响应函数，在使用createTrackbar函数时，其调用的函数可以不用写参数，甚至
	//括号都不用写，但是其调用函数的实现过程中还是需要满足(int,void*)2个参数类型
	createTrackbar("spatialRad", "dst", &spatialRad, 80);
	createTrackbar("colorRad", "dst", &colorRad, 60);
	createTrackbar("maxPryLevel", "dst", &maxPryLevel, 5);
	bool bRet = false;
	ImageSegment ism(spatialRad, colorRad, maxPryLevel); 
	bRet = ism.imageSegment(src, dst, IMG_SEGMENT_MEANSHIFT, spatialRad, colorRad, maxPryLevel);

	

	//    meanshift_seg(0,0);

	imshow("src", src);
	/*char c=(char)waitKey();
	if(27==c)
	return 0;*/
	imshow("dst", dst);
	imshow("flood", src);
	waitKey();//无限等待用户交互响应
	//    while(1);//这里不能用while(1)的原因是需要等待用户的交互，而while(1)没有该功能。虽然2者都有无限等待的作用。
	return 0;
}
