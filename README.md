# Protein-Protein Interaction Extraction based on Multi-feature Fusion and Entity Enhancement
## Dependencies

- perl (For evaluating official f1 score)
- python>=3.6
- torch==1.6.0
- transformers==3.3.1

## How to run

```bash
$ python3 main.py --do_train --do_eval
```

## Official Evaluation

```bash
$ python3 official_eval.py
```

## Prediction

```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```


