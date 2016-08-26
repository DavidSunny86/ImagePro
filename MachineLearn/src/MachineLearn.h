#pragma once
#include "MachineLearnUtils.h"


typedef struct ML_BASE_FEATURE
{
	Mat featureInfo;
	int iBtnLabel;
	String sBtnLable;
	uchar ucBtnRev[2];
}Ml_Base_Feature;



typedef enum ML_CLS_TYPE
{
	LP_BTN_CLASS_ONE,
	LP_BTN_CLASS_TWO,
	LP_BTN_CLASS_THREE,
	LP_BTN_CLASS_UNREV
};



typedef struct ML_PRE_CLS_INFO
{
	String srcClsStyle;
	int    sameBtnStyleNum;
	int    iTrainNum;
	int    iTestNum;
	int    preTrainClsCurrentNum;
	int    preTestClsCurrentNum;

	double preTrainClsPre[LP_BTN_CLASS_UNREV];
	double preTestClsPre[LP_BTN_CLASS_UNREV];

}BTN_RPE_CLS_INFO;

class MachineLearn
{
public:
	MachineLearn();
	~MachineLearn();

public:

	bool m_bIsTrainSample = false;
	bool m_bIsLoadClsModel = false;
	int  m_iMlModel;
	Ptr<StatModel> m_pMlCls
};

 