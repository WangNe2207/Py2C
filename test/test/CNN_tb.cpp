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
	float* Weights = (float*)malloc(567778 * sizeof(float));
	float tmp;
	FILE* Weight;
	errno_t fp = fopen_s(&Weight,"Float_Weights.txt", "r");
	for (int i = 0; i < 567778; i++){
		fscanf_s(Weight, "%f", &tmp);
		*(Weights + i)=tmp;
	}
	fclose(Weight);
	int choose=101;
	int d=1;
	FILE* Input;	
	float* InModel = (float*)malloc((d * 1 * 101) * sizeof(float));
	fp = fopen_s(&Input,"Input.txt", "r");
	for (int i = 0; i < d * 1 * 101; i++){
		fscanf_s(Input, "%f", &tmp);
		*(InModel + i)=tmp;
	}
	fclose(Input);
	CNN(&InModel[choose],OutModel, Weights);
	std::cout << OutModel;
	return 0;
}
