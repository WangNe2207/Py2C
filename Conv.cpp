void Conv2D_0(float Input_Conv[3072],float Output_Conv[27000], float bias[30], float kernel[810]){
	loop_for_channel2D_0:
	for (int n = 0; n < 30; n++){
		loop_for_bp2D_0:
		for (int x = 0; x < 30; x++){
			loop_for_ap2D_0:
			for (int y = 0; y < 30; y++){
				float s = 0;
				loop_for_fc_0:
				for (int k = 0; k < 3; k++){
					loop_for_fb_0:
					for (int i = 0; i < 3; i++){
						loop_for_fa_0:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[3*3*3*n+3*3*k+3*i+j])*(Input_Conv[32*32*k+32*(i+x)+j+y]);}
					}
				}
				if ((s+bias[n])<0) Output_Conv[30*30*n+30*x+y]=0; else Output_Conv[30*30*n+30*x+y]=s+bias[n];
			}
		}
	}
}
void Conv2D_1(float Input_Conv[6750],float Output_Conv[2197], float bias[13], float kernel[3510]){
	loop_for_channel2D_1:
	for (int n = 0; n < 13; n++){
		loop_for_bp2D_1:
		for (int x = 0; x < 13; x++){
			loop_for_ap2D_1:
			for (int y = 0; y < 13; y++){
				float s = 0;
				loop_for_fc_1:
				for (int k = 0; k < 30; k++){
					loop_for_fb_1:
					for (int i = 0; i < 3; i++){
						loop_for_fa_1:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[30*3*3*n+3*3*k+3*i+j])*(Input_Conv[15*15*k+15*(i+x)+j+y]);}
					}
				}
				if ((s+bias[n])<0) Output_Conv[13*13*n+13*x+y]=0; else Output_Conv[13*13*n+13*x+y]=s+bias[n];
			}
		}
	}
}
