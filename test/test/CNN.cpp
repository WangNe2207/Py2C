#include "Conv.h"
#include "Pool.h"
#include "Dense.h"
void CNN(float InModel[101],float &OutModel,float Weights[567778]){
	float OutConv0[3104];
	float OutPadPool0[3168];
	float OutPool0[1568];
	float OutConv1[800];
	float OutFlatten[800];
	float OutDense0[512];
	float OutDense1[256];
	Conv1D_0(InModel,OutConv0,&Weights[160],&Weights[0]);
	Padding_Pool1D_0(OutConv0,OutPadPool0);
	Max_Pool1D_0(OutPadPool0,OutPool0);
	Conv1D_1(OutPool0,OutConv1,&Weights[25792],&Weights[192]);
	flatten(OutConv1,OutFlatten);
	Dense_0(OutFlatten,OutDense0,&Weights[435424],&Weights[25824]);
	Dense_1(OutDense0,OutDense1,&Weights[567008],&Weights[435936]);
	Dense_2(OutDense1,OutModel,&Weights[567776],&Weights[567264]);
}
