# How to Reproduce

To reproduce all figures and tables from our [technical paper](https://arxiv.org/abs/2210.07105), do the following steps.

## Collect wandb logs

These scripts collect all wandb logs into .csv files and save them into the `runs_tables` folder. 
We provide the tables, but you can recollect them.
```python
python results/get_offline_urls.py
python results/get_finetune_urls.py
```

## Collect scores

These scripts collect data from runs kept in .csv files and save evaluation scores (and regret in case of offline-to-online) 
into pickled files, which are stored in the `bin` folder. We provide the pickled data, but if you need to extract more data,
you can modify scripts for your purposes.
```python
python results/get_offline_scores.py
python results/get_finetune_scores.py
```

## Print tables

 These scripts use pickled data, print all the tables, and save all figures into the `out` directory.
```python
python results/get_offline_tables_and_plots.py
python results/get_finetune_tables_and_plots.py
```