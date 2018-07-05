#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <float.h>


double gettime(void) {
	struct timeval ttime;
	gettimeofday(&ttime , NULL);
	return ttime.tv_sec + ttime.tv_usec * 0.000001;
}


float randpval() {
	int vr = rand();
	int vm = rand()%vr;
	float r = ((float)vm)/(float)vr;
	assert(r>=0.0 && r<=1.00001);
	return r;
}


int main(int argc, char ** argv) {

	int N = atoi(argv[1]);
	int iters = 1000;

	srand(1);

	float * mVec = (float*)malloc(sizeof(float)*N);
	assert(mVec!=NULL);
	float * nVec = (float*)malloc(sizeof(float)*N);
	assert(nVec!=NULL);
	float * LVec = (float*)malloc(sizeof(float)*N);
	assert(LVec!=NULL);
	float * RVec = (float*)malloc(sizeof(float)*N);
	assert(RVec!=NULL);
	float * CVec = (float*)malloc(sizeof(float)*N);
	assert(CVec!=NULL);
	float * FVec = (float*)malloc(sizeof(float)*N);
	assert(FVec!=NULL);

	for(int i=0;i<N;i++){
		mVec[i] = (float)(2+rand()%10);
		nVec[i] = (float)(2+rand()%10);

		LVec[i] = 0.0;
		for(int j=0;j<mVec[i];j++)
			LVec[i] += randpval();

		RVec[i] = 0.0;
		for(int j=0;j<nVec[i];j++)
			RVec[i] += randpval();

		CVec[i] = 0.0;
		for(int j=0;j<mVec[i]*nVec[i];j++)
			CVec[i] += randpval();

		FVec[i] = 0.0;

		assert(mVec[i]>=2.0 && mVec[i]<=12.0);
		assert(nVec[i]>=2.0 && nVec[i]<=12.0);
		assert(LVec[i]>0.0 && LVec[i]<=1.0*mVec[i]);
		assert(RVec[i]>0.0 && RVec[i]<=1.0*nVec[i]);
		assert(CVec[i]>0.0 && CVec[i]<=1.0*mVec[i]*nVec[i]);
	}

	float maxF = 0.0f;
	double timeTotal = 0.0f;

	//	Set vectors with the needed constants.
	__m128 vec1 = _mm_set1_ps(1.0);
	__m128 vec2 = _mm_set1_ps(2.0);
	__m128 vec001 = _mm_set_ps1(0.01f);

	//	Vectors for maximum.
	__m128 maxFVec128 = _mm_set_ps1(FLT_MIN);
	__m128 compareFlags;

	//	Vector pointers to float arrays.
	__m128* LVec128 = (__m128*) LVec;
	__m128* RVec128 = (__m128*) RVec;
	__m128* mVec128 = (__m128*) mVec;
	__m128* nVec128 = (__m128*) nVec;
	__m128* CVec128 = (__m128*) CVec;
	__m128* FVec128 = (__m128*) FVec;

	for(int j=0; j < iters; ++j){
		
		double time0 = gettime();

		//	Same i needed to be used by the two for loops. 
		int i;
		for(i=0; i < N/4; ++i){

			//	Same calculations with the serial execution. Floating point operations are non-associative and their order is respected.
			__m128 num_0 = _mm_add_ps( LVec128[i], RVec128[i]);
			__m128 num_1 = _mm_div_ps( _mm_mul_ps( mVec128[i], _mm_sub_ps( mVec128[i], vec1)), vec2);
			__m128 num_2 = _mm_div_ps( _mm_mul_ps( nVec128[i], _mm_sub_ps( nVec128[i], vec1)), vec2);
			__m128 num = _mm_div_ps( num_0, _mm_add_ps( num_1, num_2));

			__m128 den_0 = _mm_sub_ps( _mm_sub_ps( CVec128[i], LVec128[i]), RVec128[i]);
			__m128 den_1 = _mm_mul_ps( mVec128[i], nVec128[i]);
			__m128 den = _mm_div_ps( den_0, den_1);

			FVec128[i] = _mm_div_ps(num, _mm_add_ps(den, vec001));

			//	Compare verticaly and keep larger floats in a vector.
			compareFlags = _mm_cmpgt_ps( FVec128[i], maxFVec128);
			maxFVec128 = _mm_or_ps(_mm_and_ps( compareFlags, FVec128[i]),  _mm_andnot_ps(compareFlags, maxFVec128));
		}

		//	Horizontal max. Look bottom for a more detailed explanation.
		__m128 tempMax = _mm_max_ps(maxFVec128, _mm_shuffle_ps(maxFVec128, maxFVec128, _MM_SHUFFLE(0,0,3,2)));
		maxF = _mm_cvtss_f32(_mm_max_ps(tempMax, _mm_shuffle_ps(tempMax, tempMax, _MM_SHUFFLE(0,0,0,1))));

		//	Edge cases.
		for (i = i*4; i < N; ++i){
			float num_0 = LVec[i]+RVec[i];
			float num_1 = mVec[i]*(mVec[i]-1.0)/2.0;
			float num_2 = nVec[i]*(nVec[i]-1.0)/2.0;
			float num = num_0/(num_1+num_2);

			float den_0 = CVec[i]-LVec[i]-RVec[i];
			float den_1 = mVec[i]*nVec[i];
			float den = den_0/den_1;

			FVec[i] = num/(den+0.01f);

			maxF = FVec[i]>maxF?FVec[i]:maxF;
		}

		double time1 = gettime();
		timeTotal += time1-time0;
	}

	printf("Time %f Max %f\n", timeTotal/iters, maxF);

	//	Cleanup.
	free(mVec);
	free(nVec);
	free(LVec);
	free(RVec);
	free(CVec);
	free(FVec);
}

/*	Horizontal max is among the most expensive operations of the code. Needs a lot of attention.
	Removed variables to lower access to memory. Raised speedup by 5%.

static inline float horizontal_max_Vec4(__m128 x) {
	__m128 max1 = _mm_shuffle_ps(x, x, _MM_SHUFFLE(0,0,3,2));
	__m128 max2 = _mm_max_ps(x, max1);
	__m128 max3 = _mm_shuffle_ps(max2, max2, _MM_SHUFFLE(0,0,0,1));
	__m128 max4 = _mm_max_ps(max2, max3);
	float result = _mm_cvtss_f32(max4);
	return result;
}
*/