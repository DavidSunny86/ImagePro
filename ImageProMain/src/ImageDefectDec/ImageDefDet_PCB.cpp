#include "ImageDefDet_PCB.h"

ImageDefDet_PCB::ImageDefDet_PCB()
{}



cv::Mat ImageDefDet_PCB::image_Process(Mat& srcMat,Mat& mask)
{
	cv::Mat gray_input;
	cv::Mat gray_reference;
	
	cv::cvtColor(srcMat, gray_input, CV_RGB2GRAY);
	cv::cvtColor(m_modelImage, gray_reference, CV_RGB2GRAY);

	cv::equalizeHist(gray_input, gray_input);
	cv::equalizeHist(gray_reference, gray_reference);

	auto threshold = 150;
	cv::threshold(gray_input, gray_input, threshold, 255, cv::THRESH_BINARY);
	cv::threshold(gray_reference, gray_reference, threshold, 255, cv::THRESH_BINARY);

	auto strel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::morphologyEx(gray_input, gray_input, cv::MORPH_CLOSE, strel);
	cv::morphologyEx(gray_reference, gray_reference, cv::MORPH_CLOSE, strel);

	//XOR to detect differences
	cv::Mat xor_result;
	cv::bitwise_xor(gray_input, gray_reference, xor_result);
	//Get rid of noise
	cv::medianBlur(xor_result, xor_result, 3);

	//If there are non-zero pixels
	if (cv::countNonZero(xor_result) > 1)
	{
		//Outline the defects in red.

		auto output = srcMat.clone();
		//First color the defect outlines black.
		cv::Mat defect_mask = 255 - xor_result;
		cv::cvtColor(defect_mask, defect_mask, CV_GRAY2RGB);
		cv::bitwise_and(output, defect_mask, output);

		//Change the white pixels in the outline to red.
		auto color_xor = cv::Mat();
		auto dilation_strel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::dilate(xor_result, color_xor, dilation_strel, cv::Point(-1, -1));
		cv::cvtColor(color_xor, color_xor, CV_GRAY2RGB);
		for (auto pixel = color_xor.begin<cv::Vec3b>(); pixel != color_xor.end<cv::Vec3b>(); pixel++)
		{
			if ((*pixel)[0] > 0
				|| (*pixel)[1] > 0
				|| (*pixel)[2] > 0)
			{
				*pixel = cv::Vec3b(0, 0, 255);
			}
		}
		//Blend the masked output and the red outline.
		cv::addWeighted(output, 0.4, color_xor, 1.0, 0, output);
		return output;
	}

	return srcMat;
}

ImageDefDet_PCB::~ImageDefDet_PCB()
{}


