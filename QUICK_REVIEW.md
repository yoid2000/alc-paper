
The dump from was most recently tested on Python 3.12.4.

Install packages with `python -m pip install -r requirements.txt`.

## Commands for reviewer to test that measurement code works

The code that runs the attacks which are measured using ours and prior approaches are in directory `compare` (Section 6 of the paper).

```
cd compare
```

All code is in `compare.py`. The options and syntax can be displayed with:

```
python compare.py --help
```

The code is designed to run in SLURM (see for instance `slurm_script_strong`). As such, the attack measurement commands include an index into the `jobs.json` configuration file, and can be run as follows. Note that each of the following commands takes several minutes to run:

```
python compare.py attack 1093
python compare.py attack 3685
python compare.py --weak attack 1093
python compare.py --weak attack 3685
```

`python compare.py attack 1093` executes the configuration in `jobs.json` at index 1093. This is for the `adult.parquet` dataset (in `original_data_parquet`), for the prior method, using the strongly anonymized dataset in `strong_data_parquet`, and stores the results in `work_files_prior_strong/adult.1093`.  

`python compare.py attack 3685` is the same thing for index 3685, which is the same strongly anonymized data, but using our method, and storing the results in `work_files_strong/adult.3685`.

`python -w compare.py attack 1093` and `python -w compare.py attack 3685` are the same two configurations, but this time using the weakly anonymized data. They are stored in `work_files_prior_weak/adult.1093` and `work_files_weak/adult.3685` respectively.

The two strong and two weak attacks give comparable results between our approach and the prior approach.

```
python compare.py gather
```

`python compare.py gather` collects the work_files files and collates them into four results files, `all_secret_known_prior_strong.parquet`, `all_secret_known_prior_weak.parquet`, `all_secret_known_strong.parquet`, `nd all_secret_known_weak.parquet`.


```
python compare.py plot
```

`python compare.py plot` reads in the results files and generates the plots and tables in directory `plots`. When run with the results of the four above attacks, the plots are incomplete and not interesting. To replicate the figures in the paper, you can copy the contents of the `completed_results` directory into the `compare` directory, and run `python compare.py plot` again.

# Commands for reviewer to test that the dependency code works

The code that generates the dependency tests (overfitting) is in directory `dependence` (Section 3.7).

```
cd dependence
```

Each of the following commands generates a dependence measure for one dataset and column, for different overfitting settings. Each takes a few minutes to run. (This is designed to run with SLURM.)

```
python dependence.py measure 0
python dependence.py measure 5
python dependence.py measure 10
python dependence.py measure 15
```

Each run creates a json file in directory `dependence_results` containing the results of the run.

```
python dependence.py gather
```

This reads in the json files and generates `dependence_results.parquet` in the `dependence_results`.

```
python dependence.py plot
```

This reads in `dependence_results.parquet` and uses it to generate the plots (also placed in `dependence_results`). Note that plots will crash if at least the above 4 measures are not included (i.e. one each of default, overfit1, overfit2, and overfit3).

The plots using only the above four generated measures are not interesting. To generate the plot used in the paper, copy `completed_results/dependence_results.parquet` to `dependence_results` and run `python dependence.py plot`.

# Commands for reviewer to test that the composite plots code works

This code runs the plots regarding the composite scores used in Section 4 and the appendix.

```
cd prc_alc
```

The following generates the plots for Section 4.

```
python plots.py
```

The following generates the Fbeta plots for the appendix.

```
python fbeta.py
```
