#include "Conv.h"
#include "Pool.h"
#include "Dense.h"
void CNN(float InModel[27000],float &OutModel,float Weights[71919]){
	float OutConv2D0[27000];
	float OutPool0[6750];
	float OutConv2D1[2197];
	float OutPool1[78];
	float OutFlatten[468];
	float OutDense0[120];
	float OutDense1[86];
	//Conv2D_0(&InModel[0],OutConv2D0,&Weights[810],&Weights[0]);
	Max_Pool2D_0(&InModel[0],OutPool0);
	//Conv2D_1(OutPool0,OutConv2D1,&Weights[4350],&Weights[840]);
	//Max_Pool2D_1(OutConv2D1,OutPool1);
	//flatten(OutPool1,OutFlatten);
	//Dense_0(OutFlatten,OutDense0,&Weights[60523],&Weights[4363]);
	//Dense_1(OutDense0,OutDense1,&Weights[70963],&Weights[60643]);
	//Dense_2(OutDense1,OutModel,&Weights[71909],&Weights[71049]);
}
