#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <math.h>
#include<time.h>
#include "io_png.h"
#define DTYPE int
#define H 10
#define W 10
#define K 5
#define N 13
#define K2 2
#define N2 6
#define KK (K*K)
#define CROP (N2-K2)
#define PATCH_SIZE 24
#define SIZE_H_PAD  (H+2*N2)
#define SIZE_W_PAD  (W+2*N2)
//size_h_ref = H + N2 - K2
#define SIZE_H_REF (H + N2 - K2)  
//size_w_ref = W + N2 - K2
#define SIZE_W_REF (W + N2 - K2) 
#define Height 384
#define Width 512
#ifndef max
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif
const int dx_patch[24] = { -4,-2,0,2,4,-4,-2,0,2,4,-4,-2,2,4,-4,-2,0,2,4,-4,-2,0,2,4 };
const int dy_patch[24] = { -4,-4,-4,-4,-4,-2,-2,-2,-2,-2,0,0,0,0,2,2,2,2,2,4,4,4,4,4 };
const float threshold = 3*2500*KK;
void do_loop(float mat_out[H][W], float mat_in_pad[SIZE_H_PAD][SIZE_W_PAD], float mat_sqdiff[SIZE_H_REF][SIZE_W_REF])
{
	int i, j, x, y;
	float sqdiff_current;
	//#pragma EXPLORE_CONSTRAINT SAME = sqdiff_current
	float sqdiff_prev;
	//#pragma EXPLORE_CONSTRAINT SAME = sqdiff_current
	float sqdiff_save;
	//#pragma EXPLORE_CONSTRAINT SAME = mat_sqdiff
	float tmp_dist;
	float pre_exp;
	//#pragma EXPLORE_FIX W={8} I={2}
	float weight;
	//#pragma EXPLORE_CONSTRAINT SAME = mat_in_pad
	float denoised;
	float dis[SIZE_H_PAD][SIZE_W_PAD] = { 0.0f };
	float mat_out_nlm[H][W];
	float mat_acc_weight[H][W];
	float sigma2 = 900.0f;
	int count = 0;
	// Initialization
	for (i = 0; i < H; i++)
	{
		for (j = 0; j < W; j++)
		{
			mat_out_nlm[i][j] = 0;
			mat_acc_weight[i][j] = 0;
		}
	}
	for (x = -4; x <= -4; x = x + 2)
		for (y = -4; y <= -4; y = y + 2)

			if (!(x == 0 && y == 0)) {

				for (i = CROP; i < SIZE_H_REF + CROP; i++)
				{
					for (j = CROP; j < SIZE_W_REF + CROP; j++)
					{
						// Calculate square difference (and its prefix sum)
						sqdiff_current = (float)((mat_in_pad[i][j] - mat_in_pad[i + y][j + x]) * (mat_in_pad[i][j] - mat_in_pad[i + y][j + x]));
						if (i == CROP && j == CROP) {
							sqdiff_save = sqdiff_current;
							sqdiff_prev = sqdiff_current;
						}
						else if (i == CROP && j > CROP) {
							sqdiff_save = sqdiff_current + sqdiff_prev;
							sqdiff_prev = sqdiff_current + sqdiff_prev;
						}
						else if (i > CROP && j == CROP) {
							sqdiff_save = sqdiff_current + mat_sqdiff[i - CROP - 1][j - CROP];
							sqdiff_prev = sqdiff_current;
						}
						else if (i > CROP && j > CROP) {
							sqdiff_save = sqdiff_current + sqdiff_prev + mat_sqdiff[i - CROP - 1][j - CROP];
							sqdiff_prev = sqdiff_current + sqdiff_prev;
						}
						mat_sqdiff[i - CROP][j - CROP] = sqdiff_save;

						// Calculate mat_denoised and mat_acc_weight
						if ((i >= CROP + K - 1) && (j >= CROP + K - 1))
						{
							if (i == CROP + K - 1 && j == CROP + K - 1) {
								tmp_dist = mat_sqdiff[i - CROP][j - CROP];
							}
							else if (i > CROP + K - 1 && j == CROP + K - 1) {
								tmp_dist = mat_sqdiff[i - CROP][j - CROP] - mat_sqdiff[i - K - CROP][j - CROP];
							}
							else if (i == CROP + K - 1 && j > CROP + K - 1) {
								tmp_dist = mat_sqdiff[i - CROP][j - CROP] - mat_sqdiff[i - CROP][j - K - CROP];
							}
							else if (i > CROP + K - 1 && j > CROP + K - 1) {
								tmp_dist = mat_sqdiff[i - CROP][j - CROP] - mat_sqdiff[i - K - CROP][j - CROP] - mat_sqdiff[i - CROP][j - K - CROP] + mat_sqdiff[i - K - CROP][j - K - CROP];
							}
							
							//dis[i - K2 + y][j - K2 + x] = tmp_dist;
							printf("(%d,%d):%f; ", (i - K2 + y), (j - K2 + x), tmp_dist);
							/*pre_exp = tmp_dist - 2 * sigma2 * KK;
							if (pre_exp <= 0)
								weight = 1;
							else
								weight = 0;*/

								// Compute and accumulate denoised pixels
								//denoised = weight * mat_in_pad[i - K2 + y][j - K2 + x];
								//mat_out_nlm[i - CROP - K + 1][j - CROP - K + 1] += denoised;
								// Update accumulated weights
								//mat_acc_weight[i - CROP - K + 1][j - CROP - K + 1] += weight;
								//printf("(%d,%d): %f; ", (i - CROP - K + 1), (j - CROP - K + 1), mat_acc_weight[i - CROP - K + 1][j - CROP - K + 1]);
							/*for (int i = 0; i < SIZE_H_REF; i++)
								for (int j = 0; j < SIZE_W_REF; j++)
									printf("(%d,%d): %f; ", (i), (j), mat_sqdiff[i][j]);*/


						}
					}
				}
			}


	// Initialization
	//for (i = 0; i < H; i++)
	//{
	//	for (j = 0; j < W; j++)
	//	{
	//		mat_out_nlm[i][j] = 0;
	//		mat_acc_weight[i][j] = 0;
	//	}
	//}

	//for (int k = 0; k < PATCH_SIZE; k++)
	//{
	//	tmp_dist = 0;
	//	sqdiff_prev = 0;
	//	sqdiff_save = 0;
	//	for (int i = CROP; i < SIZE_H_REF + CROP; i++)
	//	{
	//		for (int j = CROP; j < SIZE_W_REF + CROP; j++)
	//		{
	//			sqdiff_current = (double)((mat_in_pad[i][j] - mat_in_pad[i + dy_patch[k]][j + dx_patch[k]]) * (mat_in_pad[i][j] - mat_in_pad[i + dy_patch[k]][j + dx_patch[k]]));

	//			if (i == CROP && j == CROP) {
	//				sqdiff_save = sqdiff_current;
	//				sqdiff_prev = sqdiff_current;
	//			}
	//			else if (i == CROP && j > CROP) {
	//				sqdiff_save = sqdiff_current + sqdiff_prev;
	//				sqdiff_prev = sqdiff_current + sqdiff_prev;
	//			}
	//			else if (i > CROP && j == CROP) {
	//				sqdiff_save = sqdiff_current + mat_sqdiff[i - CROP - 1][j - CROP];
	//				sqdiff_prev = sqdiff_current;
	//			}
	//			else if (i > CROP && j > CROP) {
	//				sqdiff_save = sqdiff_current + sqdiff_prev + mat_sqdiff[i - CROP - 1][j - CROP];
	//				sqdiff_prev = sqdiff_current + sqdiff_prev;
	//			}
	//			mat_sqdiff[i - CROP][j - CROP] = sqdiff_save;

	//			/// Calculate mat_denoised and mat_acc_weight
	//			if ((i >= CROP + K - 1) && (j >= CROP + K - 1))
	//			{
	//				if (i == CROP + K - 1 && j == CROP + K - 1) {
	//					tmp_dist = mat_sqdiff[i - CROP][j - CROP];
	//				}
	//				else if (i > CROP + K - 1 && j == CROP + K - 1) {
	//					tmp_dist = mat_sqdiff[i - CROP][j - CROP] - mat_sqdiff[i - K - CROP][j - CROP];
	//				}
	//				else if (i == CROP + K - 1 && j > CROP + K - 1) {
	//					tmp_dist = mat_sqdiff[i - CROP][j - CROP] - mat_sqdiff[i - CROP][j - K - CROP];
	//				}
	//				else if (i > CROP + K - 1 && j > CROP + K - 1) {
	//					tmp_dist = mat_sqdiff[i - CROP][j - CROP] - mat_sqdiff[i - K - CROP][j - CROP] - mat_sqdiff[i - CROP][j - K - CROP] + mat_sqdiff[i - K - CROP][j - K - CROP];
	//				}
	//				// Compute and accumulate denoised pixels
	//				pre_exp = tmp_dist - 2 * sigma2 * KK;
	//				if (pre_exp <= 0)
	//					weight = 1;
	//				else
	//					weight = 0;
	//				denoised = weight * mat_in_pad[i - K2 + dy_patch[k]][j - K2 + dx_patch[k]];
	//				mat_out_nlm[i - CROP - K + 1][j - CROP - K + 1] = mat_out_nlm[i - CROP - K + 1][j - CROP - K + 1] + denoised;
	//				printf("(%d,%d): %f; ", (i - CROP - K + 1), (i - CROP - K + 1), mat_out_nlm[i - CROP - K + 1][j - CROP - K + 1]);
	//			}

	//		}
	//	}
	//}
}

void pad_matrix_symetry(float orig[H][W], float padded[SIZE_H_PAD][SIZE_W_PAD]) {
	// just for chls=1;
	int w = SIZE_W_PAD;
	int h = SIZE_H_PAD;

	//! Center of the image
	for (unsigned i = 0; i < H; i++)
		for (unsigned j = 0; j < W; j++)
			padded[N2 + i][N2 + j] = orig[i][j];

	//! Top and bottom
	for (unsigned j = 0; j < w; j++)
		for (unsigned i = 0; i < N2; i++)
		{
			padded[i][j] = padded[2 * N2 - i - 1][j];
			padded[h - i - 1][j] = padded[h - 2 * N2 + i][j];
		}

	//! Right and left

	for (unsigned i = 0; i < h; i++)
	{
		for (unsigned j = 0; j < N2; j++)
		{
			padded[i][j] = padded[i][2 * N2 - j - 1];
			padded[i][w - j - 1] = padded[i][w - 2 * N2 + j];
		}
	}
}

int main()
{
	float Image[Height][Width];
	float img[H][W];
	float mat_sqdiff[SIZE_H_REF][SIZE_W_REF];
	float pad[SIZE_H_PAD][SIZE_W_PAD];
	char input_img[32];
	char output_img[32];
	sprintf(input_img, "test.png");

	size_t nx, ny, nc;
	float* mat_in = NULL;
	mat_in = read_png_f32(input_img, &nx, &ny, &nc);
	if (!mat_in) {
		printf("error :: %s not found  or not a correct png image \n", input_img);
		exit(-1);
	}

	int i, j;
	for (i = 0; i < Height; i++)
	{
		for (j = 0; j < Width; j++)
		{
			Image[i][j] = mat_in[j + i * Width];
		}
	}
	for (i = 0; i < H; i++)
	{
		for (j = 0; j < W; j++)
		{
			img[i][j] = Image[i][j];
			//printf("%f ", img[i][j]
		}
	}

	pad_matrix_symetry(img, pad);
	do_loop(img, pad, mat_sqdiff);
	return 0;
	


	/*for (i = 0; i < SIZE_H_PAD; i++)
		for (j = 0; j < SIZE_W_PAD; j++) {
			printf("%f ", pad[i][j]);
		}
	printf("\n");
	return 0;
*/
	//test
	/*int b[SIZE_H_PAD][SIZE_W_PAD] = { 0 };
	int i, j;
	pad_matrix_symetry(a, b);
	for (i = 0; i < SIZE_H_PAD; i++)
		for (j = 0; j < SIZE_W_PAD; j++) {
			printf("%d ", b[i][j]);
		}
		printf("\n");

	}
	printf("\n");
	return 0;

	/*int i, j;
	float img[H][W] = { {1,2,3,4,5,6,7},{1,2,3,4,5,6,7},{1,2,3,4,5,6,7},{1,2,3,4,5,6,7},{1,2,3,4,5,6,7},{1,2,3,4,5,6,7},{1,2,3,4,5,6,7} };
	float padded[SIZE_H_PAD][SIZE_W_PAD];

	pad_matrix_symetry(img, padded);

	for (i = 0; i < SIZE_H_PAD; i++)
		for (j = 0; j < SIZE_W_PAD; j++) {
			printf("%f ", padded[i][j]);
		}
	printf("\n");
	return 0;*/
}