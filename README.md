# Protein-Protein Interaction Extraction based on Multi-feature Fusion and Entity Enhancement

## Model Architecture

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68673458-1b090d00-0597-11ea-96b1-7c1453e6edbb.png" />  
</p>

## Dependencies

- perl (For evaluating official f1 score)
- python>=3.6
- torch==1.6.0
- transformers==3.3.1

## How to run

```bash
$ python3 main.py --do_train --do_eval
```

- Prediction will be written on `proposed_answers.txt` in `eval` directory.

## Official Evaluation

```bash
$ python3 official_eval.py
# macro-averaged F1 = 88.29%
```

- Evaluate based on the official evaluation perl script.
  - MACRO-averaged f1 score (except `Other` relation)
- You can see the detailed result on `result.txt` in `eval` directory.

## Prediction

```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```

## References

- [Semeval 2010 Task 8 Dataset](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?sort=name&layout=list&num=50)
- [Semeval 2010 Task 8 Paper](https://www.aclweb.org/anthology/S10-1006.pdf)
- [NLP-progress Relation Extraction](http://nlpprogress.com/english/relationship_extraction.html)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [https://github.com/wang-h/bert-relation-classification](https://github.com/wang-h/bert-relation-classification)
