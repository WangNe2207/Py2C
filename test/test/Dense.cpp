void Dense_0(float input_Dense[800],float output_Dense[512],float bias[512],float weight[409600]){
	loop_for_a_Dense_0:
	for (int i = 0; i < 512; i++){
		float s=0;
		loop_for_b_Dense_0:
		for (int j = 0; j < 800; j++){
			s+=input_Dense[j]*weight[j*512+i];
		}
		if ((s+bias[i])<0) output_Dense[i]=0; else output_Dense[i]=s+bias[i];
	}
}
void Dense_1(float input_Dense[512],float output_Dense[256],float bias[256],float weight[131072]){
	loop_for_a_Dense_1:
	for (int i = 0; i < 256; i++){
		float s=0;
		loop_for_b_Dense_1:
		for (int j = 0; j < 512; j++){
			s+=input_Dense[j]*weight[j*256+i];
		}
		if ((s+bias[i])<0) output_Dense[i]=0; else output_Dense[i]=s+bias[i];
	}
}
void Dense_2(float input_Dense[256],float &output_Dense,float bias[2],float weight[512]){
	float out_Dense[2];
	loop_for_a_Dense_2:
	for (int i = 0; i < 2; i++){
		float s=0;
		loop_for_b_Dense_2:
		for (int j = 0; j < 256; j++){
			s+=input_Dense[j]*weight[j*2+i];
		}
		out_Dense[i]=s+bias[i];
	}
	float maxindex=out_Dense[0];
	output_Dense = 0;
	loop_detect:
	for (int i=1; i<2; i++){
		if (out_Dense[i]>maxindex) {
			maxindex=out_Dense[i];
			output_Dense=i;
		}
	}
}
