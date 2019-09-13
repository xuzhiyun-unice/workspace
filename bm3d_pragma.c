#define __GECOS_TYPE_EXPL__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "io_png.h"

#ifndef __GECOS_TYPE_EXPL__
#include<time.h>
#include <float.h>
#include <stdint.h>
#endif

#ifndef __GECOS_TYPE_EXPL__
//----------------float active or commented
//#define EXP_T float
//----------------ac_fixed active or commented
#define NBBIT 20
#define NBINT 10
#define EXP_T ac_fixed<NBBIT,NBINT,1,AC_TRN,AC_WRAP>
#endif

#ifdef __GECOS_TYPE_EXPL__
#define EXP_T float
#endif

#define H 384
#define W 512
#define chnls 1
#define K 8
#define N 21
#define K2 4
#define N2 10
#define pHard 3 //step
#define kHard 8   //kHard = (tau_2D_hard == BIOR || sigma < 40.f ? 8 : 12); //! Must be a power of 2 if tau_2D_hard == BIOR
#define kHard_2 (kHard*kHard)
#define kWien 8  // kWien = (tau_2D_wien == BIOR || sigma < 40.f ? 8 : 12); //! Must be a power of 2 if tau_2D_wien == BIOR
#define kWien_2 (kWien*kWien)
#define NHard 16 //MAX NUMBLE BLOCK step1 , Must be a power of 2
#define NWien 32 //MAX NUMBLE BLOCK step2 , Must be a power of 2
#define KK (K*K)
#define SIZE_H_PAD  (H+2*N2)
#define SIZE_W_PAD  (W+2*N2)
#define row_index_size (2+(SIZE_H_PAD-N-K)/3)  //moving size of search window in H
#define column_index_size (2+(SIZE_W_PAD-N-K)/3) //moving size of search window in W

#pragma EXPLORE_FIX W={11} I={10} //val
EXP_T val_256 = 256.0; //pour normaliser pixel

#pragma EXPLORE_FIX W={8} I={7} //val
EXP_T val_32 = 32.0; // pour normaliser weight_table

#pragma EXPLORE_FIX W={12..22} I={4} //pixel
static EXP_T table_2D[N * SIZE_W_PAD][K][K] = { 0.0f };

#pragma EXPLORE_FIX W={17..22} I={8} //pixel
static EXP_T group_3D[NHard * KK] = { 0.0f };

#pragma EXPLORE_FIX W={18..22} I={9}
	static EXP_T table_2D_est[N * SIZE_W_PAD][K][K] = { 0.0f };//used for step2

#pragma EXPLORE_FIX W={18..22} I={9} //pixel
	static EXP_T group_3D_2[NWien * KK] = { 0.0f };//used for step2

#pragma EXPLORE_FIX W={18..22} I={9} //pixel
	static EXP_T group_3D_est[NWien * KK];//used for step2

#pragma EXPLORE_FIX W={17..22} I={8} //pixel
static EXP_T hadamard_tmp[NHard];

#pragma EXPLORE_FIX W={18..22} I={9} //pixel
	static EXP_T wien_tmp[NWien] = { 0.0f };//used for step2

#pragma EXPLORE_FIX W={10..17} I={2} //coeff
	static EXP_T weight_table[column_index_size] = { 0.0f };

#pragma EXPLORE_FIX  W={9..17} I={1} // I={1}//0-1
EXP_T T[K][K] =
{ { 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f},
{ 0.490393f, 0.415735f, 0.277785f, 0.097545f, -0.097545f, -0.277785f, -0.415735f, -0.490393f },
{ 0.461940f, 0.191342f, -0.191342f, -0.461940f, -0.461940f, -0.191342f, 0.191342f, 0.461940f },
{ 0.415735f, -0.097545f, -0.490393f, -0.277785f, 0.277785f, 0.490393f, 0.097545f, -0.415735f },
{ 0.353553f, -0.353553f, -0.353553f, 0.353553f, 0.353553f, -0.353553f, -0.353553f, 0.353553f },
{ 0.277785f, -0.490393f, 0.097545f, 0.415735f, -0.415735f, -0.097545f, 0.490393f, -0.277785f },
{ 0.191342f, -0.461940f, 0.461940f, -0.191342f, -0.191342f, 0.461940f, -0.461940f, 0.191342f },
{ 0.097545f, -0.277785f, 0.415735f, -0.490393f, 0.490393f, -0.415735f, 0.277785f, -0.097545f } };
#pragma EXPLORE_FIX  W={9..17} I={1} // I={1}//0-1
EXP_T Tinv[K][K] =
{ { 0.353553f, 0.490393f, 0.461940f, 0.415735f, 0.353553f, 0.277785f, 0.191342f, 0.097545f},
{ 0.353553f, 0.415735f, 0.191342f,-0.097545f,-0.353553f,-0.490393f,-0.461940f, -0.277785f },
{ 0.353553f, 0.277785f,-0.191342f,-0.490393f,-0.353553f, 0.097545f, 0.461940f,  0.415735f },
{ 0.353553f, 0.097545f,-0.461940f,-0.277785f, 0.353553f, 0.415735f,-0.191342f, -0.490393f },
{ 0.353553f,-0.097545f,-0.461940f, 0.277785f, 0.353553f,-0.415735f,-0.191342f,  0.490393f },
{ 0.353553f,-0.277785f,-0.191342f, 0.490393f,-0.353553f,-0.097545f, 0.461940f, -0.415735f },
{ 0.353553f,-0.415735f, 0.191342f, 0.097545f,-0.353553f, 0.490393f,-0.461940f,  0.277785f },
{ 0.353553f,-0.490393f, 0.461940f,-0.415735f, 0.353553f,-0.277785f, 0.191342f, -0.097545f } };
/**
 * @brief Look for the closest power of 2 number
 *
 * @param n: number
 *
 * @return the closest power of 2 lower or equal to n
 **/
int closest_power_of_2(
	const int n
) {
	int r = 1;
	while (r * 2 <= n)
		r *= 2;

	return r;
}

/**
 * @brief Initialize a set of indices.
 *
 * @param ind_set: will contain the set of indices;
 * @param max_size: indices can't go over this size;
 * @param boundary : boundary;
 * @param step: step between two indices.
 *
 * @return none.
 **/
void ind_initialize(
	int* ind_set
	, const int max_size
	, const int boundary
	, const int step
) {
	int ind = boundary;
	int count = 0;
	while (ind < max_size - boundary)
	{
		ind_set[count] = ind;
		ind += step;
		count++;
	}
	if (ind_set[count - 1] < max_size - boundary - 1) {
		ind_set[count] = (max_size - boundary - 1);
	}
}

/**
 * @brief matrix_multiplication, only for matrix with dim = k*k
 *
 * @param A : dim must be K*K
 * @param B : dim must be K*K
 * @param R : result R = A*B
 *
 **/
void matrix_multiplication(EXP_T A[K][K], EXP_T B[K][K], EXP_T R[K][K])
{
	int i, j, n;
#pragma EXPLORE_CONSTRAINT SAME = R
	EXP_T tmp;
	for (i = 0; i < K; i++) {
		for (n = 0; n < K; n++) {
			tmp = 0;
			for (j = 0; j < K; j++) {
				tmp += A[i][j] * B[j][n];
			}
			R[i][n] = tmp;
		}
	}
}

void matrix_multiplication2(EXP_T A[K][K], EXP_T B[K][K], EXP_T R[K][K])
{
	int i, j, n;
#pragma EXPLORE_CONSTRAINT SAME = R
	EXP_T tmp;
	for (i = 0; i < K; i++) {
		for (n = 0; n < K; n++) {
			tmp = 0;
			for (j = 0; j < K; j++) {
				tmp += A[i][j] * B[j][n];
			}
			R[i][n] = tmp;
		}
	}
}
void matrix_multiplication3(EXP_T A[K][K], EXP_T B[K][K], EXP_T R[K][K])
{
	int i, j, n;
#pragma EXPLORE_CONSTRAINT SAME = R
	EXP_T tmp;
	for (i = 0; i < K; i++) {
		for (n = 0; n < K; n++) {
			tmp = 0;
			for (j = 0; j < K; j++) {
				tmp += A[i][j] * B[j][n];
			}
			R[i][n] = tmp;
		}
	}
}
void matrix_multiplication4(EXP_T A[K][K], EXP_T B[K][K], EXP_T R[K][K])
{
	int i, j, n;
#pragma EXPLORE_CONSTRAINT SAME = R
	EXP_T tmp;
	for (i = 0; i < K; i++) {
		for (n = 0; n < K; n++) {
			tmp = 0;
			for (j = 0; j < K; j++) {
				tmp += A[i][j] * B[j][n];
			}
			R[i][n] = tmp;
		}
	}
}
void matrix_multiplication5(EXP_T A[K][K], EXP_T B[K][K], EXP_T R[K][K])
{
	int i, j, n;
#pragma EXPLORE_CONSTRAINT SAME = R
	EXP_T tmp;
	for (i = 0; i < K; i++) {
		for (n = 0; n < K; n++) {
			tmp = 0;
			for (j = 0; j < K; j++) {
				tmp += A[i][j] * B[j][n];
			}
			R[i][n] = tmp;
		}
	}
}
void matrix_multiplication6(EXP_T A[K][K], EXP_T B[K][K], EXP_T R[K][K])
{
	int i, j, n;
#pragma EXPLORE_CONSTRAINT SAME = R
	EXP_T tmp;
	for (i = 0; i < K; i++) {
		for (n = 0; n < K; n++) {
			tmp = 0;
			for (j = 0; j < K; j++) {
				tmp += A[i][j] * B[j][n];
			}
			R[i][n] = tmp;
		}
	}
}

void matrix_multiplication7(EXP_T A[K][K], EXP_T B[K][K], EXP_T R[K][K])
{
	int i, j, n;
#pragma EXPLORE_CONSTRAINT SAME = R
	EXP_T tmp;
	for (i = 0; i < K; i++) {
		for (n = 0; n < K; n++) {
			tmp = 0;
			for (j = 0; j < K; j++) {
				tmp += A[i][j] * B[j][n];
			}
			R[i][n] = tmp;
		}
	}
}

void matrix_multiplication8(EXP_T A[K][K], EXP_T B[K][K], EXP_T R[K][K])
{
	int i, j, n;
#pragma EXPLORE_CONSTRAINT SAME = R
	EXP_T tmp;
	for (i = 0; i < K; i++) {
		for (n = 0; n < K; n++) {
			tmp = 0;
			for (j = 0; j < K; j++) {
				tmp += A[i][j] * B[j][n];
			}
			R[i][n] = tmp;
		}
	}
}
/**
 * @brief 2d DCT transform, only for matrix with dim = k*k
 *
 * @param A : mat_in, dim must be K*K
   @param flag : flag==0 dect_2d; flag==1 inverse dec_2d
 *
 **/


 //void dct_2d(
 //#pragma EXPLORE_CONSTRAINT SAME = table_2D
 //	EXP_T A[K][K],
 //	int flag)
 //{
 //	// If A is an n x n matrix, then the following are true:
 //	//   T*A    == dct(A),  T'*A   == idct(A)
 //	//   T*A*T' == dct2(A), T'*A*T == idct2(A)
 //#pragma EXPLORE_CONSTRAINT SAME = table_2D
 //	EXP_T res1[K][K] = { 0.0f };
 //#pragma EXPLORE_CONSTRAINT SAME = table_2D
 //	EXP_T res2[K][K] = { 0.0f };
 //
 //	if (flag == 0)
 //	{
 //		matrix_multiplication(T, A, res1);
 //		matrix_multiplication2(res1, Tinv, res2); //T*A*T' == dct2(A)
 //	}
 //	if (flag == 1)
 //	{
 //		matrix_multiplication3(Tinv, A, res1);
 //		matrix_multiplication4(res1, T, res2); //T'*A*T == idct2(A)
 //	}
 //	for (int i = 0; i < K; ++i) {
 //		for (int j = 0; j < K; ++j) {
 //			A[i][j] = res2[i][j];
 //		}
 //	}
 //}

void dct_2d(
#pragma EXPLORE_CONSTRAINT SAME = table_2D
	EXP_T A[K][K]
)
{
	// If A is an n x n matrix, then the following are true:
	//   T*A    == dct(A),  T'*A   == idct(A)
	//   T*A*T' == dct2(A), T'*A*T == idct2(A)
#pragma EXPLORE_CONSTRAINT SAME = table_2D
	EXP_T res1[K][K] = { 0.0f };
#pragma EXPLORE_CONSTRAINT SAME = table_2D
	EXP_T res2[K][K] = { 0.0f };

	matrix_multiplication(T, A, res1);
	matrix_multiplication2(res1, Tinv, res2); //T*A*T' == dct2(A)

	for (int i = 0; i < K; ++i) {
		for (int j = 0; j < K; ++j) {
			A[i][j] = res2[i][j];
		}
	}
}

void dct_2d2(
#pragma EXPLORE_CONSTRAINT SAME = table_2D_est
	EXP_T A[K][K]
)
{
	// If A is an n x n matrix, then the following are true:
	//   T*A    == dct(A),  T'*A   == idct(A)
	//   T*A*T' == dct2(A), T'*A*T == idct2(A)
#pragma EXPLORE_CONSTRAINT SAME = table_2D_est
	EXP_T res1[K][K] = { 0.0f };
#pragma EXPLORE_CONSTRAINT SAME = table_2D_est
	EXP_T res2[K][K] = { 0.0f };

	matrix_multiplication5(T, A, res1);
	matrix_multiplication6(res1, Tinv, res2); //T*A*T' == dct2(A)

	for (int i = 0; i < K; ++i) {
		for (int j = 0; j < K; ++j) {
			A[i][j] = res2[i][j];
		}
	}
}

void dct_2d_inv(
#pragma EXPLORE_CONSTRAINT SAME = group_3D
	EXP_T A[K][K]
)
{
	// If A is an n x n matrix, then the following are true:
	//   T*A    == dct(A),  T'*A   == idct(A)
	//   T*A*T' == dct2(A), T'*A*T == idct2(A)
#pragma EXPLORE_CONSTRAINT SAME = group_3D
	EXP_T res1[K][K] = { 0.0f };
#pragma EXPLORE_CONSTRAINT SAME = group_3D
	EXP_T res2[K][K] = { 0.0f };

	matrix_multiplication3(Tinv, A, res1);
	matrix_multiplication4(res1, T, res2); //T'*A*T == idct2(A)

	for (int i = 0; i < K; ++i) {
		for (int j = 0; j < K; ++j) {
			A[i][j] = res2[i][j];
		}
	}
}

void dct_2d_inv_2(
#pragma EXPLORE_CONSTRAINT SAME = group_3D_2
	EXP_T A[K][K]
)
{
	// If A is an n x n matrix, then the following are true:
	//   T*A    == dct(A),  T'*A   == idct(A)
	//   T*A*T' == dct2(A), T'*A*T == idct2(A)
#pragma EXPLORE_CONSTRAINT SAME = group_3D_2
	EXP_T res1[K][K] = { 0.0f };
#pragma EXPLORE_CONSTRAINT SAME = group_3D_2
	EXP_T res2[K][K] = { 0.0f };

	matrix_multiplication7(Tinv, A, res1);
	matrix_multiplication8(res1, T, res2); //T'*A*T == idct2(A)

	for (int i = 0; i < K; ++i) {
		for (int j = 0; j < K; ++j) {
			A[i][j] = res2[i][j];
		}
	}
}


/**
 * @brief Add boundaries by symetry
 *
 * @param orig : image to symetrize
 * @param padded : will contain img with symetrized boundaries
 *
 * @return none.
 **/
void pad_matrix_symetry(EXP_T orig[H][W], EXP_T padded[SIZE_H_PAD][SIZE_W_PAD]) {

	//! Center of the image
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			padded[N2 + i][N2 + j] = orig[i][j];
		}
	}

	//! Top and bottom
	for (int j = 0; j < SIZE_W_PAD; j++) {
		for (int i = 0; i < N2; i++)
		{
			padded[i][j] = padded[2 * N2 - i - 1][j];
			padded[SIZE_H_PAD - i - 1][j] = padded[SIZE_H_PAD - 2 * N2 + i][j];
		}
	}

	//! Right and left

	for (int i = 0; i < SIZE_H_PAD; i++)
	{
		for (int j = 0; j < N2; j++)
		{
			padded[i][j] = padded[i][2 * N2 - j - 1];
			padded[i][SIZE_W_PAD - j - 1] = padded[i][SIZE_W_PAD - 2 * N2 + j];
		}
	}

}


void pad_matrix_symetry2(EXP_T orig[H][W], EXP_T padded[SIZE_H_PAD][SIZE_W_PAD]) {

	//! Center of the image
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			padded[N2 + i][N2 + j] = orig[i][j];
		}
	}

	//! Top and bottom
	for (int j = 0; j < SIZE_W_PAD; j++) {
		for (int i = 0; i < N2; i++)
		{
			padded[i][j] = padded[2 * N2 - i - 1][j];
			padded[SIZE_H_PAD - i - 1][j] = padded[SIZE_H_PAD - 2 * N2 + i][j];
		}
	}

	//! Right and left

	for (int i = 0; i < SIZE_H_PAD; i++)
	{
		for (int j = 0; j < N2; j++)
		{
			padded[i][j] = padded[i][2 * N2 - j - 1];
			padded[i][SIZE_W_PAD - j - 1] = padded[i][SIZE_W_PAD - 2 * N2 + j];
		}
	}

}




void precompute_BM(
	int patch_table[SIZE_H_PAD * SIZE_W_PAD],
	EXP_T img[SIZE_H_PAD][SIZE_W_PAD] //
	, int row_ind[row_index_size]
	, int column_ind[column_index_size]
	, int(*index_h)[SIZE_H_PAD * SIZE_W_PAD]
	, int(*index_w)[SIZE_H_PAD * SIZE_W_PAD]
	, const int width  //pad ???
	, const int height
	, const int kHW  //K
	, const int NHW   //?????
	, const int nHW   //N2
	, const int pHW   //??
	, const EXP_T    tauMatch
) {
	int i, j, n, k, di, dj, p, q;
#pragma EXPLORE_FIX W={13..22} I={5} //coeff
	const EXP_T val_K = K;
#pragma EXPLORE_FIX W={12..22} I={4} //coeff
	const EXP_T threshold = tauMatch; //7.38
	//const EXP_T threshold = 40000.0f;
#pragma EXPLORE_FIX W={16..22} I={7} //mat+
	EXP_T diff_table[SIZE_H_PAD][SIZE_W_PAD] = { 0.0f };
#pragma EXPLORE_FIX W={16..22} I={7} //mat+
	static EXP_T  sum_table[SIZE_H_PAD * SIZE_W_PAD][N2 + 1][N];
#pragma EXPLORE_FIX W={16..22} I={7} //mat+
	EXP_T tmp;
	//Initialization
	for (i = 0; i < (SIZE_H_PAD * SIZE_W_PAD); i++) {

		for (j = 0; j < N2 + 1; j++)
		{
			for (int n = 0; n < N; n++)
			{
				sum_table[i][j][n] = 2 * threshold;
				//printf("%f; ",sum_table[i][j][n]);
			}
		}
	}

	for (int di = 0; di <= nHW; di++) {
		for (int dj = 0; dj < N; dj++)
		{
			const int dk = (int)(di * width + dj) - (int)nHW;
			const int ddk = di * N + dj;

			//! Process the image containing the square distance between pixels
			for (i = nHW; i < height - nHW; i++)
			{
				for (k = nHW; k < width - nHW; k++) {
					diff_table[i][k] = (img[i + di][k + dj - nHW] - img[i][k]) * (img[i + di][k + dj - nHW] - img[i][k]);
				}

			}

			//! Compute the sum for each patches, using the method of the integral images
			const int dn = nHW * width + nHW;
			//! 1st patch, top left corner
			tmp = 0.0f;
			for (int p = 0; p < kHW; p++)
			{
				int pq = p * width + dn;
				for (int q = 0; q < kHW; q++) {
					tmp += diff_table[p + nHW][q + nHW];
				}
			}
			sum_table[dn][di][dj] = tmp;

			//! 1st row, top
			for (int j = nHW + 1; j < width - nHW; j++)
			{
				const int ind = nHW * width + j - 1;
				tmp = sum_table[ind][di][dj];
				for (int p = 0; p < kHW; p++) {
					tmp += diff_table[nHW + p][j - 1 + kHW] - diff_table[nHW + p][j - 1];
				}
				sum_table[ind + 1][di][dj] = tmp;
				//printf("(%d,%d,%d): %f; ", (ind + 1), (di), (dj), sum_table[ind + 1][di][dj]);
			}

			//! General case
			for (int i = nHW + 1; i < height - nHW; i++)
			{
				const int ind = (i - 1) * width + nHW;
				tmp = sum_table[ind][di][dj];
				//! 1st column, left
				for (int q = 0; q < kHW; q++) {
					tmp += diff_table[i - 1 + kHW][nHW + q] - diff_table[i - 1][nHW + q];
				}
				sum_table[ind + width][di][dj] = tmp;

				//! Other columns
				int k = i * width + nHW + 1;
				int pq = (i + kHW - 1) * width + kHW - 1 + nHW + 1;
				for (int j = nHW + 1; j < width - nHW; j++, k++, pq++)
				{
					sum_table[k][di][dj] =
						sum_table[k - 1][di][dj]
						+ sum_table[k - width][di][dj]
						- sum_table[k - 1 - width][di][dj]
						+ diff_table[i + kHW - 1][kHW + j - 1]
						- diff_table[i + kHW - 1][j - 1]
						- diff_table[i - 1][j + kHW - 1]
						+ diff_table[i - 1][j - 1];
					//printf("(%d,%d,%d): %f; ", (k), (di), (dj), sum_table[k][di][dj]);
				}
			}
		}
	}

	//! Precompute Bloc Matching
	for (int ind_i = 0; ind_i < row_index_size; ind_i++)
		//for (int ind_i = 0; ind_i < 1; ind_i++)
	{
		for (int ind_j = 0; ind_j < column_index_size; ind_j++)
			//for (int ind_j = 0; ind_j < 1; ind_j++)
		{
			//! Initialization
			const int  k_r = row_ind[ind_i] * width + column_ind[ind_j];
			int BlockCount = 0;


#pragma EXPLORE_FIX W={16..22} I={7} //mat+
			EXP_T table_distance[NWien] = { 0.0f };//???????? ???pair(??????????

			//! Threshold distances in order to keep similar patches
			for (int dj = -(int)nHW; dj <= (int)nHW; dj++)
			{
				for (int di = 0; di <= (int)nHW; di++) {
					if (sum_table[k_r][di][dj + nHW] < threshold) {
						if (BlockCount < NHW)
						{
							table_distance[BlockCount] = sum_table[k_r][di][dj + nHW];
							index_h[BlockCount][k_r] = row_ind[ind_i] + di;
							index_w[BlockCount][k_r] = column_ind[ind_j] + dj;
							BlockCount++;
						}
						else if (BlockCount == NHW)//sort block by value of distance
						{

							int ind_tmp1;
							int ind_tmp2;
							for (int p = 0; p < BlockCount - 1; p++) {
								for (int q = 0; q < BlockCount - 1; q++)
								{
									if (table_distance[q] > table_distance[q + 1])
									{
										tmp = table_distance[q];
										table_distance[q] = table_distance[q + 1];
										table_distance[q + 1] = tmp;
										ind_tmp1 = index_h[q][k_r];
										index_h[q][k_r] = index_h[q + 1][k_r];
										index_h[q + 1][k_r] = ind_tmp1;
										ind_tmp2 = index_w[q][k_r];
										index_w[q][k_r] = index_w[q + 1][k_r];
										index_w[q + 1][k_r] = ind_tmp2;
									}
								}
							}
							{		if ((sum_table[k_r][di][dj + nHW]) < table_distance[BlockCount - 1])
							{
								table_distance[BlockCount - 1] = sum_table[k_r][di][dj + nHW];
								index_h[BlockCount - 1][k_r] = row_ind[ind_i] + di;
								index_w[BlockCount - 1][k_r] = column_ind[ind_j] + dj;
							}
							}
						}
					}
				}
				for (int di = -(int)nHW; di < 0; di++) {
					if (sum_table[k_r][-di][-dj + nHW] < threshold) {
						if (BlockCount < NHW)
						{
							table_distance[BlockCount] = sum_table[k_r + di * width + dj][-di][-dj + nHW];
							index_h[BlockCount][k_r] = row_ind[ind_i] + di;
							index_w[BlockCount][k_r] = column_ind[ind_j] + dj;
							BlockCount++;
						}
						else if (BlockCount == NHW)
						{
							int ind_tmp1;
							int ind_tmp2;
							for (int p = 0; p < BlockCount - 1; p++) {
								for (int q = 0; q < BlockCount - 1; q++)
								{
									if (table_distance[q] > table_distance[q + 1])
									{
										tmp = table_distance[q];
										table_distance[q] = table_distance[q + 1];
										table_distance[q + 1] = tmp;
										ind_tmp1 = index_h[q][k_r];
										index_h[q][k_r] = index_h[q + 1][k_r];
										index_h[q + 1][k_r] = ind_tmp1;
										ind_tmp2 = index_w[q][k_r];
										index_w[q][k_r] = index_w[q + 1][k_r];
										index_w[q + 1][k_r] = ind_tmp2;
									}
								}
							}
							if ((sum_table[k_r + di * width + dj][-di][-dj + nHW]) < table_distance[BlockCount - 1])
							{
								table_distance[BlockCount - 1] = sum_table[k_r + di * width + dj][-di][-dj + nHW];
								index_h[BlockCount - 1][k_r] = row_ind[ind_i] + di;
								index_w[BlockCount - 1][k_r] = column_ind[ind_j] + dj;
							}
						}
					}
				}
			}
			//Number of Blocks must be power of 2;
			if (BlockCount < NHW) {
				//if (BlockCount == 1)
				//{
				//	printf("problem size ,%d \n", k_r);
				//}
				BlockCount = closest_power_of_2(BlockCount);
			}
			patch_table[k_r] = BlockCount; //get number of block
		}
	}
}


/**
 * @brief Precompute a 2D DCT transform on all patches contained in
 *        a part of the image.
 *
 * @param table_2D : will contain the 2d DCT transform for all
 *        chosen patches;
 * @param img : image on which the 2d DCT will be processed;
 * @param i_r: current index of the reference patches;
 * @param pHard: step,space in pixels between two references patches;
 * @param i_min (resp. i_max) : minimum (resp. maximum) value
 *        for i_r. In this case the whole 2d transform is applied
 *        on every patches. Otherwise the precomputed 2d DCT is re-used
 *        without processing it.
 **/
void dct_2d_process(

	EXP_T table_2D[N * SIZE_W_PAD][K][K],
#pragma EXPLORE_FIX W={9} I={1}  //pixel
	EXP_T img[SIZE_H_PAD][SIZE_W_PAD],
	int i_r
	, int i_min
	, int i_max) {
	int i, j, k, p, q;
	//! If i_r == ns, then we have to process all DCT
	if (i_r == i_min || i_r == i_max)
	{
		//! Allocating Memory
		//EXP_T vec[N * SIZE_W_PAD][K][K] = { 0.0f };
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < SIZE_W_PAD - K; j++) {
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						table_2D[(i * SIZE_W_PAD + j)][p][q] =
							img[(i_r + i - N2 + p)][j + q];
					}
				}
				dct_2d(table_2D[(i * SIZE_W_PAD + j)]); //! Process of all DCTs,flag=0;
			}
		}
	}
	else
	{
		int ds = pHard * SIZE_W_PAD;
		//! Re-use of DCT already processe
		for (int i = 0; i < N - pHard; i++) {
			for (int j = 0; j < SIZE_W_PAD - K; j++) {
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						table_2D[(i * SIZE_W_PAD + j)][p][q] =
							table_2D[(i * SIZE_W_PAD + j) + ds][p][q];
					}
				}
			}
		}
		//! Compute the new DCT
		for (int i = 0; i < pHard; i++) {
			for (int j = 0; j < SIZE_W_PAD - K; j++) {
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						table_2D[(i + N - pHard) * SIZE_W_PAD + j][p][q] =
							img[(p + i + N - pHard + i_r - N2)][j + q];
					}
				}
				dct_2d(table_2D[(i + N - pHard) * SIZE_W_PAD + j]); //! Process of new DCTs,flag=0;
			}
		}
	}
}



void dct_2d_process_2(

	EXP_T table_2D[N * SIZE_W_PAD][K][K],
#pragma EXPLORE_FIX W={9} I={1}  //pixel
	EXP_T img[SIZE_H_PAD][SIZE_W_PAD],
	int i_r
	, int i_min
	, int i_max) {
	int i, j, k, p, q;
	//! If i_r == ns, then we have to process all DCT
	if (i_r == i_min || i_r == i_max)
	{
		//! Allocating Memory
		//EXP_T vec[N * SIZE_W_PAD][K][K] = { 0.0f };
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < SIZE_W_PAD - K; j++) {
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						table_2D[(i * SIZE_W_PAD + j)][p][q] =
							img[(i_r + i - N2 + p)][j + q];
					}
				}
				dct_2d2(table_2D[(i * SIZE_W_PAD + j)]); //! Process of all DCTs,flag=0;
			}
		}
	}
	else
	{
		int ds = pHard * SIZE_W_PAD;
		//! Re-use of DCT already processe
		for (int i = 0; i < N - pHard; i++) {
			for (int j = 0; j < SIZE_W_PAD - K; j++) {
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						table_2D[(i * SIZE_W_PAD + j)][p][q] =
							table_2D[(i * SIZE_W_PAD + j) + ds][p][q];
					}
				}
			}
		}
		//! Compute the new DCT
		for (int i = 0; i < pHard; i++) {
			for (int j = 0; j < SIZE_W_PAD - K; j++) {
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						table_2D[(i + N - pHard) * SIZE_W_PAD + j][p][q] =
							img[(p + i + N - pHard + i_r - N2)][j + q];
					}
				}
				dct_2d2(table_2D[(i + N - pHard) * SIZE_W_PAD + j]); //! Process of new DCTs,flag=0;
			}
		}
	}
}


void dct_2d_processinv(
	EXP_T(*group_3D_table)[K][K],
	int NHW
) {
	//! 2D dct inverse

	for (int i = 0; i < column_index_size; i++)
	{
		for (int n = 0; n < NHW; n++) {
			dct_2d_inv(group_3D_table[n + NHard * i]); //inverse dct_2d

		}
	}
}

void dct_2d_processinv_2 //used ofr step 2
(
	EXP_T(*group_3D_table)[K][K],
	int NHW
) {
	//! 2D dct inverse

	for (int i = 0; i < column_index_size; i++)
	{
		for (int n = 0; n < NHW; n++) {
			dct_2d_inv_2(group_3D_table[n + NWien * i]);
		}
	}
}






/**
 * @brief Apply Welsh-Hadamard transform on vec (non normalized !!)
 *
 * @param vec: vector on which a Hadamard transform will be applied.
 *        It will contain the transform at the end;
 * @param tmp: must have the same size as vec. Used for convenience;
 * @param number, d: the Hadamard transform will be applied on vec[d] -> vec[d + number].
 *        number must be a power of 2!!!!
 *
 * @return numberone.
 **/
void hadamard_transform_2(
#pragma EXPLORE_CONSTRAINT SAME = group_3D
	EXP_T * vec,
#pragma EXPLORE_CONSTRAINT SAME = hadamard_tmp
	EXP_T * tmp,
	const int number,
	const int D
) {
	if (number == 1) {
		return;
	}
	else if (number == 2)
	{
		const EXP_T a = vec[D + 0];
		const EXP_T b = vec[D + 1];
		vec[D + 0] = a + b;
		vec[D + 1] = a - b;
	}
	else
	{
		const int n = number / 2;
		for (int k = 0; k < n; k++)
		{
			const EXP_T a = vec[D + 2 * k];
			const EXP_T b = vec[D + 2 * k + 1];
			vec[D + k] = a + b;
			tmp[k] = a - b;
		}
		for (int k = 0; k < n; k++) {
			vec[D + n + k] = tmp[k];
		}
		hadamard_transform_2(vec, tmp, n, D);
		hadamard_transform_2(vec, tmp, n, D + n);
	}
}

void hadamard_transform_3(
#pragma EXPLORE_CONSTRAINT SAME = group_3D_2
	EXP_T * vec,
#pragma EXPLORE_CONSTRAINT SAME = wien_tmp
	EXP_T * tmp,
	const int number,
	const int D
) {
	if (number == 1) {
		return;
	}
	else if (number == 2)
	{
		const EXP_T a = vec[D + 0];
		const EXP_T b = vec[D + 1];
		vec[D + 0] = a + b;
		vec[D + 1] = a - b;
	}
	else
	{
		const int n = number / 2;
		for (int k = 0; k < n; k++)
		{
			const EXP_T a = vec[D + 2 * k];
			const EXP_T b = vec[D + 2 * k + 1];
			vec[D + k] = a + b;
			tmp[k] = a - b;
		}
		for (int k = 0; k < n; k++) {
			vec[D + n + k] = tmp[k];
		}
		hadamard_transform_3(vec, tmp, n, D);
		hadamard_transform_3(vec, tmp, n, D + n);
	}
}

void hadamard_transform_4(
#pragma EXPLORE_CONSTRAINT SAME = group_3D_2
	EXP_T * vec,
#pragma EXPLORE_CONSTRAINT SAME = wien_tmp
	EXP_T * tmp,
	const int number,
	const int D
) {
	if (number == 1) {
		return;
	}
	else if (number == 2)
	{
		const EXP_T a = vec[D + 0];
		const EXP_T b = vec[D + 1];
		vec[D + 0] = a + b;
		vec[D + 1] = a - b;
	}
	else
	{
		const int n = number / 2;
		for (int k = 0; k < n; k++)
		{
			const EXP_T a = vec[D + 2 * k];
			const EXP_T b = vec[D + 2 * k + 1];
			vec[D + k] = a + b;
			tmp[k] = a - b;
		}
		for (int k = 0; k < n; k++) {
			vec[D + n + k] = tmp[k];
		}
		hadamard_transform_3(vec, tmp, n, D);
		hadamard_transform_3(vec, tmp, n, D + n);
	}
}

void hadamard_transform(
#pragma EXPLORE_CONSTRAINT SAME = group_3D
	EXP_T * vec,
#pragma EXPLORE_CONSTRAINT SAME = hadamard_tmp
	EXP_T * tmp,
	const int number,
	const int D
) {
	if (number == 1) {
		return;
	}
	else if (number == 2)
	{
		const EXP_T a = vec[D + 0];
		const EXP_T b = vec[D + 1];
		vec[D + 0] = a + b;
		vec[D + 1] = a - b;
	}
	else
	{
		const int n = number / 2;
		for (int k = 0; k < n; k++)
		{
			const EXP_T a = vec[D + 2 * k];
			const EXP_T b = vec[D + 2 * k + 1];
			vec[D + k] = a + b;
			tmp[k] = a - b;
		}
		for (int k = 0; k < n; k++) {
			vec[D + n + k] = tmp[k];
		}

		hadamard_transform_2(vec, tmp, n, D);
		hadamard_transform_2(vec, tmp, n, D + n);
	}
}




void ht_filtering_hadamard(

	EXP_T group_3D[NHard * KK],

	EXP_T tmp[NHard],
	const int nSx_r,
	EXP_T sigma,
	const EXP_T lambdaHard3D,
#pragma EXPLORE_FIX W={10..17} I={2} //coeff
	EXP_T * weight_table
) {
	//! Declarations
#pragma EXPLORE_FIX W={10..14} I={2}
	EXP_T c = 1.0f;
#pragma EXPLORE_FIX W={15..22} I={6}
	EXP_T nSx_r_EXP = (EXP_T)nSx_r;
#pragma EXPLORE_FIX W={13..22} I={4} //step1:0-4;step2:0-5.6
	EXP_T coef_norm = sqrtf(nSx_r_EXP);
#pragma EXPLORE_FIX W={10..22} I={2} //0-1 coeff
	EXP_T coef = c / nSx_r_EXP;
#pragma EXPLORE_FIX W={16..22} I={6} //0-32 coeff
	EXP_T count_f = 0;
	int count = 0;
	//! Process the Welsh-Hadamard transform on the 3rd dimension
	for (int n = 0; n < kHard_2; n++) {
		hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);
	}
	//! Hard Thresholding

#pragma EXPLORE_FIX W={13..22} I={4} //coeff
	const EXP_T T = lambdaHard3D * sigma * coef_norm / val_256; //0-4
	for (int k = 0; k < kHard_2 * nSx_r; k++)
	{
		if (fabs(group_3D[k]) > T) {
			count++;
		}
		else {
			group_3D[k] = 0.0f;
		}
	}
	count_f = sqrt(count);
	*weight_table = count_f / val_32;

	//! Process of the Welsh-Hadamard inverse transform
	for (int n = 0; n < kHard_2 * chnls; n++) {
		hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);
	}

	for (int k = 0; k < nSx_r * KK; k++) {
		group_3D[k] *= coef;
		//	printf("T2: %d ", (group_3D[k]));
	}
}



/**
 * @brief Wiener filtering using Hadamard transform.
 *
 * @param group_3D : contains the 3D block built on img_noisy;
 * @param group_3D_est : contains the 3D block built on img_basic;
 * @param tmp: allocated vector used in hadamard transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param sigma : contains value of noise for each channel;
 * @param weight_table: the weighting of this 3D group for each channel;
 * @param doWeight: if true process the weighting, do nothing
 *        otherwise.
 *
 * @return none.
 **/
void wiener_filtering_hadamard(
	EXP_T group_3D[NWien * KK]
	, EXP_T group_3D_est[NWien * KK]
	, EXP_T tmp[NWien]
	, const int nSx_r
	, EXP_T sigma
	,
#pragma EXPLORE_FIX W={10..17} I={2} //coeff
	EXP_T* weight_table
) {
#pragma EXPLORE_FIX W={10..14} I={2}
	EXP_T c = 1.0f;
#pragma EXPLORE_FIX W={15..22} I={7}
	EXP_T nSx_r_EXP = (EXP_T)nSx_r;
#pragma EXPLORE_FIX W={10..22} I={2} //0-1 coeff
	EXP_T coef = c / nSx_r_EXP;
#pragma EXPLORE_FIX W={15..22} I={7} //0-45 coeff
	EXP_T count_f = 0;
#pragma EXPLORE_FIX W={17..22} I={9} //0-1 coeff
	EXP_T value = 0;
	int count = 0; //0-2048
	//! Process of the Welsh-Hadamard inverse transform
	for (int n = 0; n < kWien_2; n++) {
		hadamard_transform_4(group_3D, tmp, nSx_r, n * nSx_r);
		hadamard_transform_4(group_3D_est, tmp, nSx_r, n * nSx_r);
	}

	//! Wiener Filtering
	for (int k = 0; k < kWien_2 * nSx_r; k++)
	{
		value = group_3D_est[k] * group_3D_est[k] * coef / val_32;
		value /= (value + sigma * sigma / val_256 / val_256 / val_32);
		group_3D_est[k] = group_3D[k] * value * coef * val_256 * val_256 * val_32;
		count += value;
	}
	count_f = sqrt(count);
	*weight_table = count_f / val_32;
	//printf("T2: %f ", (weight_table));
//! Process of the Welsh-Hadamard inverse transform
	for (int n = 0; n < kWien_2; n++) {
		hadamard_transform_4(group_3D_est, tmp, nSx_r, n * nSx_r);
	}
}


/**
 * @brief Compute PSNR and RMSE between img_1 and img_2
 *
 * @param img_1 : pointer to an allocated array of pixels.
 * @param img_2 : pointer to an allocated array of pixels.
 * @param psnr  : will contain the PSNR
 * @param rmse  : will contain the RMSE
 *
 * @return EXIT_FAILURE if both images haven't the same size.
 **/
void compute_psnr(
	EXP_T img_1[H][W]
	, EXP_T img_2[H][W]
)
{
	float C1 = 20.0f;
	float C2 = 255.0f;
	float C3 = 0.0f;
	float psnr = 0.0f;
	float rmse = 0.0f;
	float HW = H * W / (float)val_256 / (float)val_256;
	float tmp = 0.0f;
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			tmp += (img_1[i][j] - img_2[i][j]) * (img_1[i][j] - img_2[i][j]);
		}
		rmse = (float)sqrtf((float)tmp / (float)HW);
		C3 = (float)log10f(C2 / rmse);
		psnr = (float)(C1 * C3);
	}
	printf("PSNR : %f \n", (float)psnr);
	printf("RMSE : %f \n", (float)rmse);
}





void bm3d_png(
#pragma EXPLORE_FIX  W={9} I={1} //8-bit intput
	EXP_T  Image_in[H][W],
#pragma EXPLORE_FIX  W={9} I={1} //8-bit intput
	EXP_T  Image_out[H][W],
#pragma EXPLORE_FIX W={14..22} I={6} //coeff
	const EXP_T sigma,
#pragma EXPLORE_FIX W={12..22} I={4} //coeff
	const EXP_T * tauMatch,
#pragma EXPLORE_FIX W={11..22} I={3} //coeff
	const EXP_T lambdaHard3D

) {
	static int index_w[NWien][SIZE_H_PAD * SIZE_W_PAD] = { 0 };//index of patch_table block
	static int index_h[NWien][SIZE_H_PAD * SIZE_W_PAD] = { 0 };//index of patch_table block

#pragma EXPLORE_FIX  W={9} I={1} //pixel
	static EXP_T pad[SIZE_H_PAD][SIZE_W_PAD];
#pragma EXPLORE_FIX  W={9} I={1} //pixel
	static EXP_T img_basic[SIZE_H_PAD][SIZE_W_PAD];
	static int patch_table[SIZE_H_PAD * SIZE_W_PAD] = { 0 }; //number of blocks 0-16;


	//! For aggregation part
#pragma EXPLORE_FIX W={16..22} I={8} //pixel out
	static EXP_T denominator[SIZE_H_PAD][SIZE_W_PAD] = { 0.0f };
#pragma EXPLORE_FIX W={16..22} I={8} //pixle out
	static EXP_T numerator[SIZE_H_PAD][SIZE_W_PAD] = { 0.0f };

#pragma EXPLORE_CONSTRAINT SAME = group_3D
	static EXP_T group_3D_table[NHard * column_index_size][K][K] = { 0.0f };



#pragma EXPLORE_CONSTRAINT SAME = group_3D_2
	static EXP_T group_3D_table_wien[NWien * column_index_size][K][K] = { 0.0f };//used for step2



#pragma EXPLORE_FIX  W={9..17} I={1} // I={1} //0-1 coeff
	EXP_T kaiser_window[K][K] =  //! Kaiser Window coefficients
	{ { 0.1924f, 0.2989f, 0.3846f, 0.4325f, 0.4325f, 0.3846f, 0.2989f, 0.1924f},
	  { 0.2989f, 0.4642f, 0.5974f, 0.6717f, 0.6717f, 0.5974f, 0.4642f, 0.2989f },
	  { 0.3846f, 0.5974f, 0.7688f, 0.8644f,0.8644f, 0.7688f, 0.5974f, 0.3846f },
	  { 0.4325f, 0.6717f, 0.8644f, 0.9718f, 0.9718f, 0.8644f,0.6717f, 0.4325f },
	  { 0.4325f, 0.6717f, 0.8644f, 0.9718f, 0.9718f, 0.8644f, 0.6717f, 0.4325f },
	  { 0.3846f, 0.5974f, 0.7688f, 0.8644f, 0.8644f, 0.7688f, 0.5974f, 0.3846f },
	  { 0.2989f, 0.4642f,0.5974f, 0.6717f, 0.6717f, 0.5974f, 0.4642f, 0.2989f },
   {0.1924f, 0.2989f, 0.3846f, 0.4325f, 0.4325f, 0.3846f, 0.2989f, 0.1924f } };



	static int row_ind[row_index_size] = { 0 };
	static int column_ind[column_index_size] = { 0 };
	ind_initialize(row_ind, SIZE_H_PAD - K + 1, N2, pHard);
	ind_initialize(column_ind, SIZE_W_PAD - K + 1, N2, pHard);

	printf("step 1...");
	pad_matrix_symetry(Image_in, pad);
	precompute_BM(patch_table, pad, row_ind, column_ind, index_h, index_w, SIZE_W_PAD, SIZE_H_PAD, K, NHard, N2, pHard, tauMatch[0]);

	//! Preprocessing of Bior table

	for (int ind_i = 0; ind_i < row_index_size; ind_i++)
	{
		const int i_r = row_ind[ind_i];
		dct_2d_process(table_2D, pad, i_r, row_ind[0], row_ind[0]);

		//! Loop on j_r
		for (int ind_j = 0; ind_j < column_index_size; ind_j++)
		{
			//! Initialization
			const int j_r = column_ind[ind_j];
			const int k_r = i_r * SIZE_W_PAD + j_r;
			//! Number of similar patches
			const int nSx_r = patch_table[k_r];
			//! Build of the 3D group

			for (int n = 0; n < nSx_r; n++)
			{
				int index = index_h[n][k_r] * SIZE_W_PAD + index_w[n][k_r] + (N2 - i_r) * SIZE_W_PAD;
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						group_3D[n + (p * K + q) * nSx_r] = table_2D[index][p][q];
					}
				}
			}
			//! HT filtering of the 3D group
			ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, sigma, lambdaHard3D, &weight_table[ind_j]);


			//! Save the 3D group. The DCT 2D inverse will be done after.
			for (int n = 0; n < nSx_r; n++) {
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						group_3D_table[n + ind_j * NHard][p][q] = group_3D[n + (p * K + q) * nSx_r];
					}
				}
			}
		} //! End of loop on j_r

		   //!  Apply 2D inverse transform
		dct_2d_processinv(group_3D_table, NHard);
		for (int ind_j = 0; ind_j < column_index_size; ind_j++)
			//for (int ind_j = 0; ind_j < 1; ind_j++)
		{
			const int j_r = column_ind[ind_j];
			const int k_r = i_r * SIZE_W_PAD + j_r;
			const int nSx_r = patch_table[k_r];
			for (int n = 0; n < nSx_r; n++)
			{
				for (int p = 0; p < kHard; p++) {
					for (int q = 0; q < kHard; q++)
					{
						numerator[(index_h[n][k_r] + p)][(index_w[n][k_r] + q)] += kaiser_window[p][q]
							* weight_table[ind_j]
							* group_3D_table[n + ind_j * NHard][p][q];
						denominator[(index_h[n][k_r] + p)][(index_w[n][k_r] + q)] += kaiser_window[p][q]
							* weight_table[ind_j];
					}
				}
			}
		}
	}//! End of loop on i_r

	//! Final reconstruction

	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			if (denominator[i + N2][j + N2] == 0) {
				printf("erro: denominator=0");
			}
			else {
				img_basic[i][j] = numerator[i + N2][j + N2] / denominator[i + N2][j + N2];
			}
			Image_out[i][j] = img_basic[i][j];
		}
	}
	printf("done. \n");

#ifndef __GECOS_TYPE_EXPL__
	printf("For image after step 1 : \n");
	size_t nx, ny, nc;
	float* mat_in = io_png_read_f32("/home/x80054656/projet/testimages/i04_float_step1.png", &nx, &ny, &nc);//utiliser image denoised en flottant comme reference
	if (!mat_in) {
		printf("error :: %s not found  or not a correct png image \n");
		exit(-1);
	}
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			Image_in[i][j] = (EXP_T)mat_in[i * W + j] / val_256;
		}
	}
	compute_psnr(Image_in, Image_out);
#endif

	//step2
	printf("step 2...");
	pad_matrix_symetry2(Image_out, img_basic);
	precompute_BM(patch_table, img_basic, row_ind, column_ind, index_h, index_w, SIZE_W_PAD, SIZE_H_PAD, K, NWien, N2, pHard, tauMatch[1]);

	for (int ind_i = 0; ind_i < row_index_size; ind_i++)
	{
		const int i_r = row_ind[ind_i];
		dct_2d_process(table_2D, pad, i_r, row_ind[0], row_ind[0]);
		dct_2d_process_2(table_2D_est, img_basic, i_r, row_ind[0], row_ind[0]);
		//! Loop on j_r
		for (int ind_j = 0; ind_j < column_index_size; ind_j++)
		{
			//! Initialization
			const int j_r = column_ind[ind_j];
			const int k_r = i_r * SIZE_W_PAD + j_r;
			//! Number of similar patches
			const int nSx_r = patch_table[k_r];
			//! Build of the 3D group
			for (int n = 0; n < nSx_r; n++)
			{
				int index = index_h[n][k_r] * SIZE_W_PAD + index_w[n][k_r] + (N2 - i_r) * SIZE_W_PAD;
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						group_3D_2[n + (p * K + q) * nSx_r] = table_2D[index][p][q];
						group_3D_est[n + (p * K + q) * nSx_r] = table_2D_est[index][p][q];
					}
				}
			}
			//! HT filtering of the 3D group
			wiener_filtering_hadamard(group_3D_2, group_3D_est, wien_tmp, nSx_r, sigma, &weight_table[ind_j]);
			//! 3D weighting using Standard Deviation

			//if (useSD)
			//	sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);

			//! Save the 3D group. The DCT 2D inverse will be done after.
			for (int n = 0; n < nSx_r; n++) {
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						group_3D_table_wien[n + ind_j * NWien][p][q] = group_3D_est[n + (p * K + q) * nSx_r];
					}
				}
			}
		} //! End of loop on j_r

		   //!  Apply 2D inverse transform
		dct_2d_processinv_2(group_3D_table_wien, NWien);
		//! Registration of the weighted estimation

//! Registration of the weighted estimation

		for (int ind_j = 0; ind_j < column_index_size; ind_j++)
		{
			const int j_r = column_ind[ind_j];
			const int k_r = i_r * SIZE_W_PAD + j_r;
			const int nSx_r = patch_table[k_r];
			for (int n = 0; n < nSx_r; n++)
			{
				for (int p = 0; p < kHard; p++) {
					for (int q = 0; q < kHard; q++)
					{
						numerator[(index_h[n][k_r] + p)][(index_w[n][k_r] + q)] += kaiser_window[p][q]
							* weight_table[ind_j]
							* group_3D_table_wien[n + ind_j * NWien][p][q];

						denominator[(index_h[n][k_r] + p)][(index_w[n][k_r] + q)] += kaiser_window[p][q]
							* weight_table[ind_j];
					}
				}
			}
		}

	}//! End of loop on i_r
	//! Final reconstruction

	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			if (denominator[i + N2][j + N2] == 0) {
				printf("erro: denominator=0");
			}
			else {
				img_basic[i][j] = numerator[i + N2][j + N2] / denominator[i + N2][j + N2];
			}
			Image_out[i][j] = img_basic[i][j];
		}
	}
	printf("done. \n");

#ifndef __GECOS_TYPE_EXPL__
	printf("For image after step 2 : \n");

	mat_in = io_png_read_f32("/home/x80054656/projet/testimages/i04_float_step2.png", &nx, &ny, &nc);//utiliser image denoised en flottant comme reference
	if (!mat_in) {
		printf("error :: %s not found  or not a correct png image \n");
		exit(-1);
	}
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			Image_in[i][j] = (EXP_T)mat_in[i * W + j] / val_256;
		}
	}
	compute_psnr(Image_in, Image_out);
#endif
}


#ifdef __GECOS_TYPE_EXPL__
#pragma MAIN_FUNC
void treat_one_imag(
#pragma EXPLORE_FIX  W={9} I={1} //8-bit intput
	EXP_T Image_in[H][W],
#pragma EXPLORE_FIX  W={9} I={1} //8-bit output
	EXP_T	Image_out[H][W],
#pragma EXPLORE_FIX W={14..22} I={6} //coeff donne
	EXP_T sigma,
#pragma EXPLORE_FIX W={12..22} I={4} //coeff donne
	EXP_T tauMatch[2],
#pragma EXPLORE_FIX W={11..22} I={3} //coeff donne
	EXP_T    lambdaHard3D
) {
	$inject(Image_in, $from_file("/home/x80054656/projet/testimages/i04_gray_01_3.png"), H, W);
	$inject(Image_out, $from_var(0), H, W);
	$inject(sigma, $from_var(10.0f));
	$inject(tauMatch[0], $from_var((7500 * K * K / 256 / 256)));
	$inject(tauMatch[1], $from_var((450 * K * K / 256 / 256)));
	$inject(lambdaHard3D, $from_var(2.125f));

	//normalize
//	for (int i = 0; i < H; i++)
//	{
//		for (int j = 0; j < W; j++)
//		{
//			Image_in[i][j] = Image_in[i][j]/val_256;
//			//Image_ref[i][j] = Image_ref[i][j]/val_256;
//		}
//	}

	bm3d_png(Image_in, Image_out, sigma, tauMatch, lambdaHard3D);

	//	//denormalize
	//	for (int i = 0; i < H; i++)
	//	{
	//		for (int j = 0; j < W; j++)
	//		{
	//			Image_out[i][j] = Image_out[i][j]*val_256;
	//
	//		}
	//	}
	$save(Image_out);
}
#endif

#ifndef __GECOS_TYPE_EXPL__
void treat_one_image(const char* infile, const char* outfile) {

	EXP_T* Image_in = (EXP_T*)malloc(sizeof(EXP_T) * H * W);
	EXP_T* Image_out = (EXP_T*)malloc(sizeof(EXP_T) * H * W);
	EXP_T sigma = 10.0f;
	EXP_T tauMatch[2] = { sqrtf(7500) ,sqrtf(400) }; // tauMatch: threshold used to determinate similarity between patches
	EXP_T lambdaHard3D = 2.1f;  //! Threshold for Hard Thresholding
	EXP_T psnr;
	EXP_T rmse;


	// read input
	size_t nx, ny, nc;
	float* mat_in = io_png_read_f32(infile, &nx, &ny, &nc);
	if (!mat_in) {
		printf("error :: %s not found  or not a correct png image \n", infile);
		exit(-1);
	}

	printf("reading image :: %s \n", infile);
	printf("image size :\n");
	printf(" - width          = %d  \n", nx);
	printf(" - height         = %d  \n", ny);
	printf(" - nb of channels = %d  \n", nc);
	for (int i = 0; i < H * W; i++) {
		Image_in[i] = (EXP_T)((EXP_T)mat_in[i] / val_256);
	}

	bm3d_png((EXP_T(*)[W])Image_in, (EXP_T(*)[W])Image_out, sigma, tauMatch, lambdaHard3D);

	for (int i = 0; i < H * W; i++) {
		mat_in[i] = (float)(Image_out[i] * val_256);
	}
	if (io_png_write_f32(outfile, mat_in, (size_t)nx, (size_t)ny, (size_t)nc) != 0) {
		printf("... failed to save png image %s", outfile);
	}
	printf("Saving png image into :: %s \n", outfile);
	free(mat_in);
	free(Image_in);
	free(Image_out);

}
#endif

#ifndef __GECOS_TYPE_EXPL__
int main(int argc, char** argv) {
	clock_t t;
	t = clock();
	treat_one_image("/home/x80054656/projet/testimages/i04_gray_01_3.png", "/home/x80054656/projet/testimages/i04_fixed_step2.png");

	t = clock() - t;
	float time_taken = ((float)t) / CLOCKS_PER_SEC; // in seconds
	printf("bm3d executed in %f seconds \n", time_taken);
	return 0;
}
#endif
