# CDCR_benchmarking

This repository contains the code used for benchmarking diverse datasets for cross-document coreference resolution (CDCR). 

The project uses a unified format to read-in datasets. 
To acquire unified CRCR datasets, please refer to this project: https://github.com/anastasia-zhukova/Diverse_CDCR_datasets

## Installation

1. **Python 3.8 required**
2. !!! Recommended to create a venv.
3. Install libraries: `pip install -r requirements.txt`
4. Set up the `experiment_config.json` and `dataset_config.json` to your needs (instructions below).
4. To execute, run `pipeline.py`.

## Config Files

This project comes with two config files. Each one can be altered to individualize the training and testing process to suit your needs.

### experiment_config.json

The experiment config is used to set up your training & testing process. 

| Field                  | Type       | Description                                                            |
|------------------------|------------|------------------------------------------------------------------------|
| train_datasets         | dictionary | Name of the datasets and true/false value to use predefined splits.    |
| test_datasets          | dictionary | Name of the datasets and true/false value to use predefined splits.    |
| dev_datasets           | dictionary | Name of the datasets and true/false value to use predefined splits.    |
| singletons             | dictionary | Whether to include singleton mentions (set per split).                 |
| model_name_train       | string     | Name of the model that is being trained.                               |
| model_name_test        | string     | Name of the model to test (none if same as training model).            |
| evaluation_granularity | dictionary | Set dataset/topic/subtopic to true to enable evaluation on that level. |
| evaluation_metrics     | dictionary | Specify which metric to use.                                           |
| metric_aggregation     | dictionary | ?                                                                      |

Example:
```json
{
	"train_datasets": [
		{
			"name": "ECBplus",    // name of the dataset (should be in-line with datasets_config.json)
			"given_split": true  // whether to use split data from datasets_config.json or the whole dataset
		}
	],
	"test_datasets": [
		{
			"name": "NewsWCL50",
			"given_split": false
		},
                {
			"name": "MEANTIME_en",
			"given_split": false
		}
	],
	"dev_datasets": [
		{
			"name": "ECBplus",
			"given_split": true
		}
	],
	"singletons": {
		"train": true,
		"test": true,
		"dev": false
	},
	"model_name_train": "experiment27_02",
	"model_name_test": null,
	"evaluation_granularity": {
		"dataset": false,
		"topic": true,
		"subtopic": true
	},
	"evaluation_metrics": {
		"old_conll": true,
		"conll_lea": false
	},
	"metric_aggregation": {
		"separate_mention_types": true,
		"combined_mention_types": true
	}
}
```

### datasets_config.json

The datasets_config defines where to pull data from to analyse.
You can add more datasets by adding more entries like the example below in the following format 
```json
{
  "datasets": [
    { "example dataset 1" },
    { "example dataset 2" },
    ...
    { "example dataset N" }
  ]
}
```

| Field                  | Type       | Description                                                               |
|------------------------|------------|---------------------------------------------------------------------------|
| name                   | string     | The name of the dataset (should be in-line with the files).               |
| conll_url              | string     | The url to retrieve the conll data from.                                  |
| mentions_url           | dictionary | A dictionary of urls to allow multiple mention files being used.          |
| split_topics           | dictionary | Set "topic_level" or "doc_level" as key to provide splitting granularity. |

Example dataset:
```json
{
  "name": "MEANTIME_en",
  "conll_url": "https://raw.githubusercontent.com/anastasia-zhukova/Diverse_CDCR_datasets/master/MEANTIME-prep/output_data/en/meantime.conll",
  "mentions_url": [
    "https://raw.githubusercontent.com/anastasia-zhukova/Diverse_CDCR_datasets/master/MEANTIME-prep/output_data/en/event_mentions.json",
    "https://raw.githubusercontent.com/anastasia-zhukova/Diverse_CDCR_datasets/master/MEANTIME-prep/output_data/en/entities_mentions.json"
  ],
  "split_topics": {
    "topic_level": {
      "train": [],
      "test": [
        "corpus_airbus",
        "corpus_apple",
        "corpus_gm",
        "corpus_stock"
      ],
      "dev": []
    }
  }
}
```

When adding a new dataset to the list, make sure the downloadable files (conll & mentions) behind the URLs are correctly formatted. 

Please refer to this repository for formatting instructions: https://github.com/anastasia-zhukova/Diverse_CDCR_datasets