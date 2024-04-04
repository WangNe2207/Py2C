void Max_Pool2D_0(float input_MaxPooling[28800], float output_MaxPooling[6750]){
	float pool = 0.0;
	float value=0.0;
	int s;
	loop_for_channel_pool_0:
	for (int z = 0; z < 30; z++){
		loop_for_weight_pool_0:
		for (int y = 0; y < 15; y++){
			s=y+y;
			pool = input_MaxPooling[32*z+s];
			value = input_MaxPooling[32*z+s+1];
			if (value > pool)
				pool=value;
			value = input_MaxPooling[32*z+s+2];
			if (value > pool) pool=value;
			output_MaxPooling[15*z+y]=pool;
		}
	}
}
void Padding_Pool2D_0(float input_Pad_Pool[27000], float output_Pad_Pool[28800]){
	loop_for_3_channel_pad_0:
	for (int c = 0; c < 30; c++)
		loop_for_channel_pad_0:
		for (int n = 0; n < 30; n++){
			loop_for_weight_pad_0:
			for (int i = 0; i < 32; i++){
				if (i < 1 || i >= 31) output_Pad_Pool[32*n*c+i]=0; else output_Pad_Pool[32*n*c+i]=input_Pad_Pool[30*n*c+i-1];
			}
		}
	}
}
void Max_Pool2D_1(float input_MaxPooling[2535], float output_MaxPooling[637]){
	float pool = 0.0;
	float value=0.0;
	int s;
	loop_for_channel_pool_1:
	for (int z = 0; z < 13; z++){
		loop_for_weight_pool_1:
		for (int y = 0; y < 7; y++){
			s=y+y;
			pool = input_MaxPooling[15*z+s];
			value = input_MaxPooling[15*z+s+1];
			if (value > pool)
				pool=value;
			value = input_MaxPooling[15*z+s+2];
			if (value > pool) pool=value;
			output_MaxPooling[7*z+y]=pool;
		}
	}
}
void Padding_Pool2D_1(float input_Pad_Pool[2197], float output_Pad_Pool[2535]){
	loop_for_3_channel_pad_1:
	for (int c = 0; c < 13; c++)
		loop_for_channel_pad_1:
		for (int n = 0; n < 13; n++){
			loop_for_weight_pad_1:
			for (int i = 0; i < 15; i++){
				if (i < 1 || i >= 14) output_Pad_Pool[15*n*c+i]=0; else output_Pad_Pool[15*n*c+i]=input_Pad_Pool[13*n*c+i-1];
			}
		}
	}
}
void flatten(float input_Flatten[91],float output_Flatten[637]){
	int hs = 0;
	loop_for_a_flatten:
	for (int i = 0; i < 7; i++){
		loop_for_c_flatten:
		for (int j = 0; j < 13; j++){
			output_Flatten[hs] = input_Flatten[7*j+i];
			hs++;
		}
	}
}
