void Padding_Conv2D_0(float input_Pad_Conv[784], float output_Pad_Conv[900]){
	loop_for_3_channel_pad_0:
	for (int c = 0; c < 1; c++){
		loop_for_channel_pad_0:
		for (int n = 0; n < 30; n++){
			loop_for_weight_pad_0:
			for (int i = 0; i < 30; i++){
				if (n < 1.0 || n >= 29.0) output_Pad_Conv[30 * 30 * c + 30 * n + i]=0;
				 else 
					if (i < 1.0 || i >= 29.0) output_Pad_Conv[30 * 30 * c + 30 * n + i]=0; else output_Pad_Conv[30 * 30 * c + 30 * n + i] = input_Pad_Conv[28 * 28 * c + 28 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_0(float Input_Conv[900],float Output_Conv[25088], float bias[32], float kernel[288]){
	loop_for_channel2D_0:
	for (int n = 0; n < 32; n++){
		loop_for_bp2D_0:
		for (int x = 0; x < 28; x++){
			loop_for_ap2D_0:
			for (int y = 0; y < 28; y++){
				float s = 0;
				loop_for_fc_0:
				for (int k = 0; k < 1; k++){
					loop_for_fb_0:
					for (int i = 0; i < 3; i++){
						loop_for_fa_0:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[1*3*3*n+3*3*k+3*i+j])*(Input_Conv[30*30*k+30*(i+x)+j+y]);}
					}
				}
				if ((s+bias[n])<0) Output_Conv[28*28*n+28*x+y]=0; else Output_Conv[28*28*n+28*x+y]=s+bias[n];
			}
		}
	}
}
#include <cmath>
 void BatchNorm2D_0(float Input_BatchNorm[25088], float Output_BatchNorm[25088], float gamma[32], float beta[32], float MovMean[32], float MovVar[32]) {
	float eps = 0.001;
	 for(int i = 0; i < 32; i++){
		for(int j = 0; j < 784; j++){
			 Output_BatchNorm[784 * i + j] = ((Input_BatchNorm[784 * i + j] - MovMean[i]) / (sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
void Padding_Conv2D_1(float input_Pad_Conv[6272], float output_Pad_Conv[8192]){
	loop_for_3_channel_pad_1:
	for (int c = 0; c < 32; c++){
		loop_for_channel_pad_1:
		for (int n = 0; n < 16; n++){
			loop_for_weight_pad_1:
			for (int i = 0; i < 16; i++){
				if (n < 1.0 || n >= 15.0) output_Pad_Conv[16 * 16 * c + 16 * n + i]=0;
				 else 
					if (i < 1.0 || i >= 15.0) output_Pad_Conv[16 * 16 * c + 16 * n + i]=0; else output_Pad_Conv[16 * 16 * c + 16 * n + i] = input_Pad_Conv[14 * 14 * c + 14 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_1(float Input_Conv[8192],float Output_Conv[6272], float bias[32], float kernel[9216]){
	loop_for_channel2D_1:
	for (int n = 0; n < 32; n++){
		loop_for_bp2D_1:
		for (int x = 0; x < 14; x++){
			loop_for_ap2D_1:
			for (int y = 0; y < 14; y++){
				float s = 0;
				loop_for_fc_1:
				for (int k = 0; k < 32; k++){
					loop_for_fb_1:
					for (int i = 0; i < 3; i++){
						loop_for_fa_1:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[32*3*3*n+3*3*k+3*i+j])*(Input_Conv[16*16*k+16*(i+x)+j+y]);}
					}
				}
				if ((s+bias[n])<0) Output_Conv[14*14*n+14*x+y]=0; else Output_Conv[14*14*n+14*x+y]=s+bias[n];
			}
		}
	}
}
#include <cmath>
 void BatchNorm2D_1(float Input_BatchNorm[6272], float Output_BatchNorm[6272], float gamma[32], float beta[32], float MovMean[32], float MovVar[32]) {
	float eps = 0.001;
	 for(int i = 0; i < 32; i++){
		for(int j = 0; j < 196; j++){
			 Output_BatchNorm[196 * i + j] = ((Input_BatchNorm[196 * i + j] - MovMean[i]) / (sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
void Padding_Conv2D_2(float input_Pad_Conv[6272], float output_Pad_Conv[8192]){
	loop_for_3_channel_pad_2:
	for (int c = 0; c < 32; c++){
		loop_for_channel_pad_2:
		for (int n = 0; n < 16; n++){
			loop_for_weight_pad_2:
			for (int i = 0; i < 16; i++){
				if (n < 1.0 || n >= 15.0) output_Pad_Conv[16 * 16 * c + 16 * n + i]=0;
				 else 
					if (i < 1.0 || i >= 15.0) output_Pad_Conv[16 * 16 * c + 16 * n + i]=0; else output_Pad_Conv[16 * 16 * c + 16 * n + i] = input_Pad_Conv[14 * 14 * c + 14 * (n - 1) + i - 1];
			}
		}
	}
}
void Conv2D_2(float Input_Conv[8192],float Output_Conv[6272], float bias[32], float kernel[9216]){
	loop_for_channel2D_2:
	for (int n = 0; n < 32; n++){
		loop_for_bp2D_2:
		for (int x = 0; x < 14; x++){
			loop_for_ap2D_2:
			for (int y = 0; y < 14; y++){
				float s = 0;
				loop_for_fc_2:
				for (int k = 0; k < 32; k++){
					loop_for_fb_2:
					for (int i = 0; i < 3; i++){
						loop_for_fa_2:
						for (int j = 0; j < 3; j++){
							s=s+(kernel[32*3*3*n+3*3*k+3*i+j])*(Input_Conv[16*16*k+16*(i+x)+j+y]);}
					}
				}
				if ((s+bias[n])<0) Output_Conv[14*14*n+14*x+y]=0; else Output_Conv[14*14*n+14*x+y]=s+bias[n];
			}
		}
	}
}
#include <cmath>
 void BatchNorm2D_2(float Input_BatchNorm[6272], float Output_BatchNorm[6272], float gamma[32], float beta[32], float MovMean[32], float MovVar[32]) {
	float eps = 0.001;
	 for(int i = 0; i < 32; i++){
		for(int j = 0; j < 196; j++){
			 Output_BatchNorm[196 * i + j] = ((Input_BatchNorm[196 * i + j] - MovMean[i]) / (sqrt(MovVar[i] + eps))) * gamma[i] + beta[i];
		}
	}
}
 void Activation0(float Input_Activation[6272], float Output_Activation[6272]){
	for (int i = 0; i < 6272; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
