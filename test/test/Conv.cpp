void Conv1D_0(float Input_Conv[101],float Output_Conv[3104], float bias[32], float kernel[160]){
	loop_for_channel_0:
	for (int n = 0; n < 32; n++){
		loop_for_ap_0:
		for (int y = 0; y < 97; y++){
			float s = 0;
			loop_for_fc_0:
			for (int k = 0; k < 1; k++){
				loop_for_fa_0:
				for (int j = 0; j < 5; j++){
					s=s+(kernel[1*5*n+5*k+j])*(Input_Conv[101*k+j+y]);}
			}
			if ((s+bias[n])<0) Output_Conv[97*n+y]=0; else Output_Conv[97*n+y]=s+bias[n];
		}
	}
}
void Conv1D_1(float Input_Conv[1568],float Output_Conv[800], float bias[32], float kernel[25600]){
	loop_for_channel_1:
	for (int n = 0; n < 32; n++){
		loop_for_ap_1:
		for (int y = 0; y < 25; y++){
			float s = 0;
			loop_for_fc_1:
			for (int k = 0; k < 32; k++){
				loop_for_fa_1:
				for (int j = 0; j < 25; j++){
					s=s+(kernel[32*25*n+25*k+j])*(Input_Conv[49*k+j+y]);}
			}
			if ((s+bias[n])<0) Output_Conv[25*n+y]=0; else Output_Conv[25*n+y]=s+bias[n];
		}
	}
}
