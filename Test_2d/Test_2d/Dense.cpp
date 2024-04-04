void Dense_0(float input_Dense[637],float output_Dense[120],float bias[120],float weight[76440]){
	loop_for_a_Dense_0:
	for (int i = 0; i < 120; i++){
		float s=0;
		loop_for_b_Dense_0:
		for (int j = 0; j < 637; j++){
			s+=input_Dense[j]*weight[j*120+i];
		}
		if ((s+bias[i])<0) output_Dense[i]=0; else output_Dense[i]=s+bias[i];
	}
}
void Dense_1(float input_Dense[120],float output_Dense[86],float bias[86],float weight[10320]){
	loop_for_a_Dense_1:
	for (int i = 0; i < 86; i++){
		float s=0;
		loop_for_b_Dense_1:
		for (int j = 0; j < 120; j++){
			s+=input_Dense[j]*weight[j*86+i];
		}
		if ((s+bias[i])<0) output_Dense[i]=0; else output_Dense[i]=s+bias[i];
	}
}
void Dense_2(float input_Dense[86],float &output_Dense,float bias[10],float weight[860]){
	float out_Dense[10];
	loop_for_a_Dense_2:
	for (int i = 0; i < 10; i++){
		float s=0;
		loop_for_b_Dense_2:
		for (int j = 0; j < 86; j++){
			s+=input_Dense[j]*weight[j*10+i];
		}
		out_Dense[i]=s+bias[i];
	}
	float maxindex=out_Dense[0];
	output_Dense = 0;
	loop_detect:
	for (int i=1; i<10; i++){
		if (out_Dense[i]>maxindex) {
			maxindex=out_Dense[i];
			output_Dense=i;
		}
	}
}
