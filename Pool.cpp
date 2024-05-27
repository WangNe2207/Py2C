void Max_Pool2D_0(float input_MaxPooling[25088], float output_MaxPooling[6272]){
	int PoolSize = 2;
	int stride = 2;
	int index = 0;
	for (int i = 0; i < 32; i++){
		index = 0;
		for (int z = 0; z < 14; z++){
			for (int y = 0; y < 14; y++){
				float max_c = -10;
				for (int h = 0; h < PoolSize; h++){
					for (int w = 0; w < PoolSize; w++){
						int Pool_index = 28 * 28 * i + 28 * h + 28 * stride * z + w + y * stride;
						float Pool_value = input_MaxPooling[Pool_index];
						if (Pool_value >= max_c) max_c = Pool_value;
					}
				}
				int outIndex = 14 * 14 * i + index;
				output_MaxPooling[outIndex] = max_c;
				index++;
			}
		}
	}
}
void Max_Pool2D_1(float input_MaxPooling[6272], float output_MaxPooling[1568]){
	int PoolSize = 2;
	int stride = 2;
	int index = 0;
	for (int i = 0; i < 32; i++){
		index = 0;
		for (int z = 0; z < 7; z++){
			for (int y = 0; y < 7; y++){
				float max_c = -10;
				for (int h = 0; h < PoolSize; h++){
					for (int w = 0; w < PoolSize; w++){
						int Pool_index = 14 * 14 * i + 14 * h + 14 * stride * z + w + y * stride;
						float Pool_value = input_MaxPooling[Pool_index];
						if (Pool_value >= max_c) max_c = Pool_value;
					}
				}
				int outIndex = 7 * 7 * i + index;
				output_MaxPooling[outIndex] = max_c;
				index++;
			}
		}
	}
}
void flatten(float input_Flatten[1568],float output_Flatten[1568]){
	int hs = 0;
	for (int i = 0; i < 7; i++){
		for (int j = 0; j < 7; j++){
			for (int k = 0; k < 32; k++){
				output_Flatten[hs] = input_Flatten[7 * i + 7 * 7 * k + j ];
				hs++;
			}
		}
	}
}
