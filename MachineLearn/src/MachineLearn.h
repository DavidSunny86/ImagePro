#pragma once
#include "MachineLearnUtils.h"
#include "ImageBaseUtils.h"
#include "opencv2/ml/ml.hpp"


typedef struct ML_BASE_FEATURE
{
	Mat    featureInfo;
	int    iLabel;
	String sLable;
	uchar  ucRev[2];
}Ml_Base_Feature;



typedef enum ML_CLS_TYPE
{
	ML_CLASS_ONE,
	ML_CLASS_TWO,
	ML_CLASS_THREE,
	ML_CLASS_UNREV
};


typedef enum ML_METHOD_TYPE
{
	ML_METHOD_ADABOOST,
	ML_METHOD_RTREES,
	ML_METHOD_MLP,
	ML_METHOD_KNN,
	ML_METHOD_SVM,
	ML_METHOD_NBAYES,
	ML_METHOD_UNREV
}ML_METHOD_TYPE;


typedef enum ML_FEATURE_TYPE
{
	ML_TEXT_FEATURE,
	ML_COLOR_FEATURE,
	ML_AVE_COLOR_FEATURE,
	ML_CCV_COLOR_FEATURE,
	ML_CCV_VECTOR_FEATURE,
	ML_CCVHSV_COLOR_FEATURE,
	ML_SHAPE_FEATURE,
	ML_SHAPECONTEXT_FEATURE,
	ML_TEXT_GLCM,
	ML_TEXT_HIST,
	ML_FEATURE_MAX

};


const int class_count = CT_THREE;
const int computerFeatureMode = ML_TEXT_FEATURE;


typedef struct ML_PRE_CLS_INFO
{
	String srcClsStyle;
	int    sameStyleNum;
	int    iTrainNum;
	int    iTestNum;
	int    preTrainClsCurrentNum;
	int    preTestClsCurrentNum;

	double preTrainClsPre[ML_CLASS_UNREV];
	double preTestClsPre[ML_CLASS_UNREV];

}ML_RPE_CLS_INFO;

class MachineLearn
{
public:
	MachineLearn();
	~MachineLearn();

public:

	bool m_bIsTrainSample = false;
	bool m_bIsLoadClsModel = false;
	int  m_iMlModel;
	Ptr<StatModel> m_pMlCls;
};

 