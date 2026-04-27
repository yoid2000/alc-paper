# alc-paper

The dump from was most recently tested on Python 3.12.4.

Install packages with `python -m pip install -r requirements.txt`.

For more information, see the `README.md` files in each of the code subdirectories.

## Version

The results in this repo are generated from anonymity_loss_coefficient version 1.0.27. This version is automatically installed from requirements.txt.

## Commands for reviewer to test that code works

```
cd compare
python compare.py -h

python compare.py attack 1093
python compare.py attack 3685
python -w compare.py attack 1093
python -w compare.py attack 3685

python compare.py gather
python compare.py plot
```

## Explanation of commands

`python compare.py -h` gives the various command options

The following four commands take about 5 minutes each to run.

`python compare.py attack 1093` executes the configuration in `jobs.json` at index 1093. (This is designed to work with SLURM.) This is for the `adult.parquet` dataset (in `original_data_parquet`), for the prior method, using the strongly anonymized dataset in `strong_data_parquet`, and stores the results in `work_files_prior_strong/adult.1093`.  

`python compare.py attack 3685` is the same thing for index 3685, which is the same strongly anonymized data, but using our method, and storing the results in `work_files_strong/adult.3685`.

`python -w compare.py attack 1093` and `python -w compare.py attack 3685` are the same two configurations, but this time using the weakly anonymized data. They are stored in `work_files_prior_weak/adult.1093` and `work_files_weak/adult.3685` respectively.

`python compare.py gather` normally collects the work_files files and collates them into four results files, `all_secret_known_prior_strong.parquet`, `all_secret_known_prior_weak.parquet`, `all_secret_known_strong.parquet`, `nd all_secret_known_weak.parquet`. We've pre-populated these with all of the final data, so this does nothing in this case.

`python compare.py plot` reads in the results files and generates the plots and tables used in the paper.

## Datasets

The datasets used in the paper can be found in directory `original_data_parquet`. The datasets in `weak_data_parquet` and `strong_data_parquet` are generated using `prep_anon.py`. 


### Code

Directory `dependence` contains the code and results for generating the plot showing the effect of dependent rows in the data.

Directory `compare` contains the code for running the attacks using both our approach and a prior approach similar to Giomi et al.'s Anonymeter.

Directory `prc_alc`contains the code generating the plots showing the behavior of the PRC and ALC measures.

Directory `anonymity_loss_coefficient` is a copy of the code used for both the ALC measure and the attacks, version 1.0.32. This code from this location is not used directly by the experimental code. Rather, the code as used here is pip installed.