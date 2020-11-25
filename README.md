# Parallel GTV for Seasonal Forecasting 

This is a parallel version of this [GTV](https://github.com/Willett-Group/gtv_forecasting) using MPI in python

### Requirements

In addition to the required dependencies of [GTV](https://github.com/Willett-Group/gtv_forecasting), the following packages are also required:
- mpi4py 
- geopandas

### Running the code in parallel
```
$ mpirun -n num_process python main.py
```
You can use the jupyter notebook gtv_plot.ipynb for visualization

## Authors

* Phong Le 

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE.md](LICENSE.md) file for details.
