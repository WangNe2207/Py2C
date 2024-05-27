#include "Conv.h"
#include "Pool.h"
#include "Dense.h"
#include <algorithm>
#include <string.h>
void addArray(float* array1, float* array2, float* result, int size) {
	for (int i = 0; i < size; i++){
		result[i] = array1[i] + array2[i];
	}
}
void CNN(float InModel[784],float &OutModel,float Weights[423434]){
	float OutPadConv0[900];
	float OutConv2D0[25088];
	float OutBatchNorm2D_0[25088];
	float OutPool0[6272];
	float skip_0[6272];
	float OutPadConv1[8192];
	float OutConv2D1[6272];
	float OutBatchNorm2D_1[6272];
	float OutPadConv2[8192];
	float OutConv2D2[6272];
	float OutBatchNorm2D_2[6272];
	float end_basicblock_0[6272];
	float OutActivation0[6272];
	float OutPool1[1568];
	float OutFlatten[1568];
	float OutDense0[256];
	Padding_Conv2D_0(&InModel[0],OutPadConv0);
	Conv2D_0(OutPadConv0,OutConv2D0,&Weights[288],&Weights[0]);
	BatchNorm2D_0(OutConv2D0,OutBatchNorm2D_0,&Weights[320],&Weights[352],&Weights[384],&Weights[416]);
	Max_Pool2D_0(OutBatchNorm2D_0,OutPool0);
	memcpy(skip_0, OutPool0, sizeof(OutPool0));
	Padding_Conv2D_1(OutPool0,OutPadConv1);
	Conv2D_1(OutPadConv1,OutConv2D1,&Weights[9664],&Weights[448]);
	BatchNorm2D_1(OutConv2D1,OutBatchNorm2D_1,&Weights[9696],&Weights[9728],&Weights[9760],&Weights[9792]);
	Padding_Conv2D_2(OutBatchNorm2D_1,OutPadConv2);
	Conv2D_2(OutPadConv2,OutConv2D2,&Weights[19040],&Weights[9824]);
	BatchNorm2D_2(OutConv2D2,OutBatchNorm2D_2,&Weights[19072],&Weights[19104],&Weights[19136],&Weights[19168]);
	addArray(OutBatchNorm2D_2, skip_0, end_basicblock_0, 6272);
	Activation0(end_basicblock_0,OutActivation0);
	Max_Pool2D_1(OutActivation0,OutPool1);
	flatten(OutPool1,OutFlatten);
	Dense_0(OutFlatten,OutDense0,&Weights[420608],&Weights[19200]);
	Dense_1(OutDense0,OutModel,&Weights[423424],&Weights[420864]);
}
