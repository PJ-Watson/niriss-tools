#ifndef WORKER_H_

#define WORKER_H_

static double max(double a, double b);

static double radius(double angle, double sma, double eps);

void worker(double* result, double* weight, int n_rows, int n_cols, int N, int high_harmonics,
		double* fss_arr, double* intens_arr, double* eps_arr, double* pa_arr,
		double* x0_arr, double* y0_arr, double* a3_arr, double* b3_arr, double* a4_arr, double* b4_arr,
		int nthreads);


#endif
