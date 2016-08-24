  #include "ImageEdge.h"


ImageEdge::ImageEdge(Mat& srcMat):m_srcMat(srcMat),m_iResult(1)
{
}

int ImageEdge:: ConnectEdge(IplImage * src)
{
	if (NULL == src)
		return 1;

	int width = src->width;
	int height = src->height;

	uchar * data = (uchar *)src->imageData;
	for (int i = 2; i < height - 2; i++)
	{
		for (int j = 2; j < width - 2; j++)
		{
			//��������ĵ�Ϊ255,�������İ�����  
			if (data[i * src->widthStep + j] == 255)
			{
				int num = 0;
				for (int k = -1; k < 2; k++)
				{
					for (int l = -1; l < 2; l++)
					{
						//������������лҶ�ֵΪ0�ĵ㣬��ȥ�Ҹõ��ʮ������  
						if (k != 0 && l != 0 && data[(i + k) * src->widthStep + j + l] == 255)
							num++;
					}
				}
				//�����������ֻ��һ������255��˵�������ĵ�Ϊ�˵㣬��������ʮ������  
				if (num == 1)
				{
					for (int k = -2; k < 3; k++)
					{
						for (int l = -2; l < 3; l++)
						{
							//����õ��ʮ����������255�ĵ㣬��õ������ĵ�֮��ĵ���Ϊ255  
							if (!(k < 2 && k > -2 && l < 2 && l > -2) && data[(i + k) * src->widthStep + j + l] == 255)
							{
								data[(i + k / 2) * src->widthStep + j + l / 2] = 255;
							}
						}
					}
				}
			}
		}
	}

	return 0;
}




ImageEdge::~ImageEdge()
{
}
