#include "jpg.h"
#include "mnist.h"
#include<limits.h>


float dist(float* v1, float* v2){
	float d=0;
	for(int i=0; i<784; i++){
		d+=(v1[i]-v2[i])*(v1[i]-v2[i]);
	}
	return d;
}

int linear_classifier(float *w, float *x){
	float d = 0;
	for(int  i=0; i<784; i++){
			d+= w[i] *x[i];
	}
	if(d>=0){
		return 1;
	}else{
		return 0;
	}
}





int main()
{

    float** images = read_mnist("train-images.idx3-ubyte");
	float* labels = read_labels("train-labels.idx1-ubyte");
	float** test_images = read_mnist("t10k-images.idx3-ubyte");
	float* test_labels = read_labels("t10k-labels.idx1-ubyte");
	float err=0;
	float *w = new float[784];


	//Partie KMEANS

		// 1)Init

	const int  K=10;
	float A [K][784];
	float B [K][784];
	int* n = new int [K];
	

	for (int i=0; i<K; i++){
		for (int j=0; j<784; j++){
			A[i][j]= (float)rand()*2/INT_MAX-1;
			B[i][j]=0;
		}
	}	

		// 2) Main loop
	
	for (int t=0; t<1000; t++){
	
			// step 1

		for (int i=0; i<K; i++){
			n[i]=0;
			for (int j=0; j<784; j++){
				B[i][j]=0;
			}
		}
	

			//step 2

		for (int i=0; i<60000; i++){
				printf ("t=%d, i=%u \n", t, i);
			float mind = -1; int gagnant =0;
			for (int k=0; k<K; k++){
				float d = dist(A[k], images [i]);
				if (d<= mind || mind == -1){
					mind = d; gagnant= k;
				}
			}

			//Accumulation des images dans les barycentres de leur cellules gagnantes
			for (int j=0; j<784; j++){
				B[gagnant][j] += images [i][j];
				n[gagnant] ++;
			}	
		}
		

			//step3

		for (int k=0; k<K; k++){
			for (int j=0; j<784; j++){
				A[k][j] = B[k][j]/n[k];
			}
		}for (int k=0; k<K; k++){
			save_jpg( A[k], 28, 28, "%u/%u.jpg", k, t);
		}
	}		
		
	










	//STEP 1: INITIALISATION	
	for(int i =0; i<784;i++){
		w[i]=(float)rand()*2/INT_MAX-1;
	}
	float gamma = 0.01;
    

	//STEP 2: LEARNING	
	for (int i=0; i<100; i+=3) {
		int prediction = linear_classifier(w, images [i]);
	int verite = (labels[i] == 1) ? 1 : -1;
	if (verite != prediction) {
		printf ("ERREUR\n");
		for (int j=0; j<784;j++){
			w[j] = w[j] + gamma*verite*images [i][j];
		}
	}
	}			


	for(int i=0; i<10000; i++) {
		int inf = linear_classifier(w, test_images[i]); 
		printf("%u\n", i);
        save_jpg(test_images[i], 28, 28, "%u/%04u.jpg", inf, i);
		if((inf ==1 && test_labels[i]!= 1)||( inf==0 && test_labels[i]== 1)){
			err++;
		}
	printf("Il y a %0.2f%% d'erreur. \n", (err*100)/i);
    }
	printf("Il y a %0.2f %% d'erreur. \n", err);
    return 0;
}

