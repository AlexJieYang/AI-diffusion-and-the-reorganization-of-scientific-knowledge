"""Run before importing NumPy/pandas in each fresh kernel."""
import os

def configure():
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['OMP_THREAD_LIMIT'] = '8'
    os.environ['MKL_DYNAMIC'] = 'FALSE'
    os.environ['ARROW_NUM_THREADS'] = '4'
    try:
        from threadpoolctl import threadpool_limits
        global _thread_limit
        _thread_limit = threadpool_limits(limits=1, user_api='blas')
    except ImportError:
        pass

configure()
