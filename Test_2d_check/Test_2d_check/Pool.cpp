void Max_Pool2D_0(float input_MaxPooling[27000], float output_MaxPooling[6750]){
	int PoolSize = 2;
	int stride = 2;
	int index = 0;
	for (int i = 0; i < 30; i++){
		index = 0;
		for (int z = 0; z < 15; z++){
			for (int y = 0; y < 15; y++){
				float max_c = 0.0;
				for (int h = 0; h < PoolSize; h++){
					for (int w = 0; w < PoolSize; w++){
						int Pool_index = 30 * 30 * i + 30 *  h + 30 * stride * z + w + y * stride;
						float Pool_value = input_MaxPooling[Pool_index];
						if (Pool_value >= max_c) max_c = Pool_value;
					}
				}
				int outIndex = 15 * 15 * i + index;
				output_MaxPooling[outIndex] = max_c;
				index++;
			}
		}
	}
}
void Max_Pool2D_1(float input_MaxPooling[2197], float output_MaxPooling[468]){
	int PoolSize = 2;
	int stride = 2;
	int index = 0;
	for (int i = 0; i < 13; i++){
		index = 0;
		for (int z = 0; z < 6; z++){
			for (int y = 0; y < 6; y++){
				float max_c = 0.0;
				for (int h = 0; h < PoolSize; h++){
					for (int w = 0; w < PoolSize; w++){
						int Pool_index = 13 * 13 * i + 13 *  h + 13 * stride * z + w + y * stride;
						float Pool_value = input_MaxPooling[Pool_index];
						if (Pool_value >= max_c) max_c = Pool_value;
					}
				}
				int outIndex = 6 * 6 * i + index;
				output_MaxPooling[outIndex] = max_c;
				index++;
			}
		}
	}
}
void flatten(float input_Flatten[468],float output_Flatten[468]){
	int hs = 0;
	for (int i = 0; i < 6; i++){
		for (int j = 0; j < 6; j++){
			for (int k = 0; k < 13; k++){
				output_Flatten[hs] = input_Flatten[6 * i + 6 * 6 * k + j ];
				hs++;
			}
		}
	}
}
