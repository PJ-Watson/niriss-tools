// Originally written by HuanLe02

#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include "worker.h"

#define PI 3.1415926535897932384626433

// MAX FUNCTION
static double max(double a, double b)
{
  if (a > b) return a;
  else return b;
}

// radius function, obtained from code of EllipseGeomtry
// https://photutils.readthedocs.io/en/stable/_modules/photutils/isophote/geometry.html#EllipseGeometry
static double radius(double angle, double sma, double eps)
{
  double val = (sma * (1. - eps) /
                sqrt(pow(((1. - eps) * cos(angle)),2) +
                        pow((sin(angle)),2)));
  return val;
}

// WORKER FUNCTION
void worker(double* result, double* weight, int n_rows, int n_cols, int N, int high_harmonics,
		double* fss_arr, double* intens_arr, double* eps_arr, double* pa_arr,
		double* x0_arr, double* y0_arr, double* a3_arr, double* b3_arr, double* a4_arr, double* b4_arr,
		int nthreads)
{
  int i;
  int n_entries = n_rows * n_cols;

  omp_set_num_threads(nthreads);

  #pragma omp parallel default(shared) private(i)
  {
    // private arrays for each thread to write, calloc auto initializes to 0.0
    double* result_priv = calloc(n_entries, sizeof(double));
    double* weight_priv = calloc(n_entries, sizeof(double));

    #pragma omp for schedule(static) nowait
    for (i = 1; i < N; i++) {
      // variables, obtained at array[index]
      double sma0 = fss_arr[i];
      double intens = intens_arr[i];
      double eps = eps_arr[i];
      double pa = pa_arr[i];
      double x0 = x0_arr[i];
      double y0 = y0_arr[i];

      // Ellipse geometry, _phi_min (defined in original code as 0.05)
      // https://photutils.readthedocs.io/en/stable/_modules/photutils/isophote/geometry.html#EllipseGeometry
      double _phi_min = 0.05;

      // scan angle
      double r = sma0;
      double phi = 0.0;
      while (phi <= 2*PI + _phi_min) {
        double harm = 0.0;
        if (high_harmonics) {
          harm = (a3_arr[i] * sin(3.0*phi) +
          	b3_arr[i] * cos(3.0*phi) +
          	a4_arr[i] * sin(4.0*phi) +
          	b4_arr[i] * cos(4.0*phi));
        }
        double x = r * cos(phi + pa) + x0;
        double y = r * sin(phi + pa) + y0;
        int i = (int) x;
        int j = (int) y;
        double fx, fy;

        if (i > 0 && i < n_cols-1 && j > 0 && j < n_rows-1) {
          // fractional deviations
          fx = x - i;
          fy = y - j;

          // transform 2D index to 1D index on a flattened array
          // ex: j_i = [j,i], j1_i = [j+1,i]
          int j_i = i + j*n_cols;
          int j_i1 = (i+1) + j*n_cols;
          int j1_i = i + (j+1)*n_cols;
          int j1_i1 = (i+1) + (j+1)*n_cols;

          // isophote contribution
          result_priv[j_i] += (intens + harm) * (1.0 - fy) * (1.0 - fx);
          result_priv[j_i1] += (intens + harm) * (1.0 - fy) * fx;
          result_priv[j1_i] += (intens + harm) * fy * (1.0 - fx);
          result_priv[j1_i1] += (intens + harm) * fy * fx;

          // fractional area contribution
          weight_priv[j_i] += (1.0 - fy) * (1.0 - fx);
          weight_priv[j_i1] += (1.0 - fy) * fx;
          weight_priv[j1_i] += fy * (1.0 - fx);
          weight_priv[j1_i1] += fy * fx;

          // step next pixel on ellipse
          phi = max((phi + 0.75/r), _phi_min);
          r = max(radius(phi, sma0, eps), 0.5);
        }
        else {
          break;
        }
      }
    }

    // write to main array, one thread at a time
    #pragma omp critical
    {
      for (int ct = 0; ct < n_entries; ct++) {
        result[ct] += result_priv[ct];
	      weight[ct] += weight_priv[ct];
      }
    }

    // free memory
    free(result_priv);
    free(weight_priv);
  }
}
