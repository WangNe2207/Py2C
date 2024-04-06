#include<iostream>
void Max_Pool2D_0(float input_MaxPooling[27000], float output_MaxPooling[6750]){
	int PoolSize = 2;
	int index = 0.0;
	for (int i = 0; i < 30; i++){
		for (int z = 0; z < 15; z++){
			for (int y = 0; y < 15; y++){
				for (int c = 0; c < 3; c++){
					float max_c = 0.0;
					for (int h = 0; h < PoolSize; h++){
						for (int w = 0; w < PoolSize; w++){
							int Pool_index = 15 * 15 * c + 15 * (h + z) + w + y;
							float Pool_value = input_MaxPooling[Pool_index];
							if (Pool_value >= max_c) max_c = Pool_value;
						}
					}
					int outIndex = 15 * 15 * c + index;
					output_MaxPooling[outIndex] = max_c;
				}
				index++;
			}
		}
	}
}
void Max_Pool2D_1(float input_MaxPooling[2197], float output_MaxPooling[468]){
	int PoolSize = 2;
	for (int i = 0; i < 13; i++){
		for (int z = 0; z < 6; z++){
			for (int y = 0; y < 6; y++){
				for (int c = 0; c < 3; c++){
					for (int h = 0; h < PoolSize; h++){
						for (int w = 0; w < PoolSize; w++){
							int Pool_index = 6 * 6 * c + 6 * (h + z) + w + y;
							float Pool_value = input_MaxPooling[Pool_index];
						}
					}
				}
			}
		}
	}
}
void flatten(float input_Flatten[78],float output_Flatten[468]){
	int hs = 0;
	loop_for_a_flatten:
	for (int i = 0; i < 6; i++){
		loop_for_c_flatten:
		for (int j = 0; j < 13; j++){
			output_Flatten[hs] = input_Flatten[6*j+i];
			hs++;
		}
	}
}
