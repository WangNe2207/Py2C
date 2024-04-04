#include "Conv.h"
#include "Pool.h"
#include "Dense.h"
void CNN(float InModel[1024],float &OutModel,float Weights[92199]){
	float OutConv2D0[27000];
	float OutPadPool0[960];
	float OutPool0[450];
	float OutConv2D1[2197];
	float OutPadPool1[195];
	float OutPool1[91];
	float OutFlatten[637];
	float OutDense0[120];
	float OutDense1[86];
	Conv2D_0(InModel,OutConv2D0,&Weights[810],&Weights[0]);
	Padding_Pool2D_0(OutConv2D0,OutPadPool0);
	Max_Pool2D_0(OutPadPool0,OutPool0);
	Conv2D_1(OutPool0,OutConv2D1,&Weights[4350],&Weights[840]);
	Padding_Pool2D_1(OutConv2D1,OutPadPool1);
	Max_Pool2D_1(OutPadPool1,OutPool1);
	flatten(OutPool1,OutFlatten);
	Dense_0(OutFlatten,OutDense0,&Weights[80803],&Weights[4363]);
	Dense_1(OutDense0,OutDense1,&Weights[91243],&Weights[80923]);
	Dense_2(OutDense1,OutModel,&Weights[92189],&Weights[91329]);
}
