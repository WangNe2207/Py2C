#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <fstream>
#include <iostream>
#include "CNN.h"
#include "Conv.h"
#include "Pool.h"
#include "Dense.h"
int main(){
	float OutModel;
	float* Weights = (float*)malloc(92199 * sizeof(float));
	float tmp;
	FILE* Weight;
	errno_t fp = fopen_s(&Weight,"Weights.txt", "r");
	for (int i = 0; i < 92199; i++){
		fscanf_s(Weight, "%f", &tmp);
		*(Weights + i)=tmp;
	}
	fclose(Weight);
	//int choose=...;
	//int d=...;
	//FILE* Input;
	//float* InModel = (float*)malloc((d * 32 * 32) * sizeof(float));
	//fp = fopen_s(&Input,"Input.txt", "r");
	//for (int i = 0; i < d * 32 * 32; i++){
		//fscanf(Input, "%f", &tmp);
		//*(InModel + i)=tmp;
	//}
	//fclose(Input);
	CNN(&InModel[choose],OutModel, Weights);
	std::cout << OutModel;
	return 0;
}
