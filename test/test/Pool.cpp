void Max_Pool1D_0(float input_MaxPooling[3168], float output_MaxPooling[1568]){
	float pool = 0.0;
	float value=0.0;
	int s;
	loop_for_channel_pool_0:
	for (int z = 0; z < 32; z++){
		loop_for_weight_pool_0:
		for (int y = 0; y < 49; y++){
			s=y+y;
			pool = input_MaxPooling[99*z+s];
			value = input_MaxPooling[99*z+s+1];
			if (value > pool)
				pool=value;
			value = input_MaxPooling[99*z+s+2];
			if (value > pool) pool=value;
			output_MaxPooling[49*z+y]=pool;
		}
	}
}
void Padding_Pool1D_0(float input_Pad_Pool[3104], float output_Pad_Pool[3168]){
	loop_for_channel_pad_0:
	for (int n = 0; n < 32; n++){
		loop_for_weight_pad_0:
		for (int i = 0; i < 99; i++){
			if (i < 1 || i >= 98) output_Pad_Pool[99*n+i]=0; else output_Pad_Pool[99*n+i]=input_Pad_Pool[97*n+i-1];
		}
	}
}
void flatten(float input_Flatten[800],float output_Flatten[800]){
	int hs = 0;
	loop_for_a_flatten:
	for (int i = 0; i < 25; i++){
		loop_for_c_flatten:
		for (int j = 0; j < 32; j++){
			output_Flatten[hs] = input_Flatten[25*j+i];
			hs++;
		}
	}
}
