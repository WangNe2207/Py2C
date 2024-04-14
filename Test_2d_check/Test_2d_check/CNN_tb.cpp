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
#define NumberOfPicture  10000
#define d 3
int main(){
	float OutModel;
	//read Weight
	float* Weights = (float*)malloc(71919 * sizeof(float));
	float tmp;
	FILE* Weight;
	errno_t fp = fopen_s(&Weight,"Float_Weights.txt", "r");
	for (int i = 0; i < 71919; i++){
		fscanf_s(Weight, "%f", &tmp);
		*(Weights + i)=tmp;
	}
	fclose(Weight);
	//read input
	FILE* Input;
	float* InModel = (float*)malloc((NumberOfPicture * d * 32 * 32) * sizeof(float));
	fp = fopen_s(&Input, "ImageTXT.txt", "r");
	for (int i = 0; i < NumberOfPicture * d * 32 * 32; i++)
	{
		fscanf_s(Input, "%f", &tmp);
		*(InModel + i) = tmp;
	}
	fclose(Input);
	//read Label
	FILE* Output;
	float* Label = (float*)malloc((NumberOfPicture) * sizeof(float));
	fp = fopen_s(&Output, "LabelTXT.txt", "r");
	for (int i = 0; i < NumberOfPicture; i++)
	{
		fscanf_s(Output, "%f", &tmp);
		*(Label + i) = tmp;
	}
	fclose(Output);
	float OutArray[NumberOfPicture] = {};
	float Image[d * 32 * 32] = {};
	for (int i = 0; i < NumberOfPicture ; i++)
	{
		int startIndex = i * d * 32 * 32;
		for (int k = 0; k < d * 32 * 32; k++)
		{
			Image[k] = *(InModel + startIndex + k);
		}
		CNN(Image, OutModel, Weights);
		OutArray[i] = OutModel;
	}
	float countTrue = 0;
	for (int i = 0; i < NumberOfPicture; i++)
	{
		int labelValue = *(Label + i);
		if (labelValue == OutArray[i])
		{
			countTrue = countTrue + 1;
		}
	}
	float accuracy = (float)((countTrue / NumberOfPicture) * 100);
	std::cout << "accuracy of Model: " << accuracy << "%\n";
	return 0;
}
