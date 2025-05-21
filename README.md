# Meta-PG

## Introduction

We propose Meta-PG, a lightweight framework for test-time knowledge injection that converts textual knowledge into plug-in parameters via meta-learned parameter generation. This allows the model to incorporate external knowledge directly into its parameters, enabling it to utilize this information during generation as if it had been learned during pre-training.

## Usage

### Installing environment

```bash
conda create -n meta_pg python=3.10.15
conda activate meta_pg
pip install -r requirements.txt
```

### Downloading datasets

For 2WikiMultihopQA: Download it from https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1. Unzip it and move the files to `data/2wikimultihopqa`.

For HotpotQA:
```bash
mkdir data/hotpotqa
cd data/hotpotqa
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
```

### Training Meta-learing Model

Go the `src` folder
```bash
cd src
```

Run training with the following command:
```bash
python Meta.py \
--peft_config_file [your_peft_config_file].json \
--train_args_file [your_train_args_file].json \
--generation_config_file [your_generation_config_file] \
--learner_config_file [your_learner_config_file].json \
--output_dir [your_output_dir]
```

Here is the meanings of arguments:
- `peft_config_file`: The config file of PEFT. Now only LoRA is supported.
- `train_args_file`: The config file of the training. Refer to the `TrainArgs` in `src/Meta.py` for the meanings of the arguments.
- `generation_config_file`: The config file of the generation in regular evaluation during training.
- `learner_config_file`: The config file of the Meta-learning model. Refer to the `LearnerConfig` in `src/Meta.py` for the meanings of the arguments.
- `output_dir`: The output directory of the model. The configs and the training log will be saved here as well.
- `dataset`: The dataset to be used for training. `mix_multi` is recommended.
- `overwrite`: Whether to overwrite the output directory if it exists.

For those config files, Seeing samples of the config files in `config/sample` folder might be helpful.

We also provide a script for finetuning the model. See `src/FT.py`. The way to run that is similar to `src/Meta.py`.

#### Potential Issues

You may occurs the following error when running the training script:
```
RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'
```

Solve it by the following steps:
- goto the env foler: `cd [your_env_folder]`. It is usually at `[your miniconda_folder]/envs/meta_pg`.
- open the file `lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py`
- Searching for `triu`, and you will find the following code:
    ```python
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
    ```
- Replace it with the following code:
    ```python
            if sequence_length != 1:
                if dtype == torch.bfloat16:
                    causal_mask = causal_mask.to(torch.float32)
                    causal_mask = torch.triu(causal_mask, diagonal=1)
                    causal_mask = causal_mask.to(device=device, dtype=torch.bfloat16)
                else:
                    causal_mask = torch.triu(causal_mask, diagonal=1)
    ```

### Evaluating the model

Go the `src` folder
```bash
cd src
```

Run evaluating with the following command:
```bash
python eval.py \
--mode [ICL|Meta|FT|FTsimAda] \
--model_name_or_path [your_model_name_or_path] \
--prediction_file [your_prediction_file]
```

Here is the meanings of arguments:
- `mode`: The method to be evaluated, including:
    - ICL: In-context learning. Answering the question using the original LLM.
    - Meta: Our method Meta_PG.
    - FT: Finetuning. Answering the question using a finetuned LLM.
    - FTsimAda: Finetuning and a simple adaptation before testing. Based on the finetuned model, adapt the model to the context before testing on the related question, by a simple one-step gradient descent on the language modeling loss on the context.
- `dataset`: The dataset to be evaluated. `mix_multi` is recommended.
- `model_name_or_path`: The path of the model to be evaluated.
- `blind_context`: If it is set, the context will not be provided in the prompt. For example, in ICL mode, if `blind_context` is set, the context is unseen by the LLM during evaluation.
- `only_related`: If it is set, only the passage that contains the supporting facts to answer the question will be provided in the context.
- `prediction_file`: The output file of the predictions.
- `evaluate_loss`: Whether to evaluate the loss on the correct answers.
- `detailed`: Whether to output the detailed scores for each question.
- `shift`: If set, evaluating the model with mismatched context-question pairs.
- `random_init`: It only take effect when `mode` is `FTsimAda`. If set, the model will be initialized with random weights, rather than the finetuned model.
- `mixed_precision`: mixed_precision mode for `Acclerator`.

#### Argument Selection of Baselines

Here is the argument selection to evaluate various baselines shown in the paper.

| Baselines     | Arguments                     |
| ------------- | ----------------------------- |
| Direct        | `--mode ICL --blind_context`  |
| Task SFT      | `--mode FT`                   |
| ICL           | `--mode ICL`                  |
| Meta-PG       | `--mode Meta --blind_context` |
| Meta-PG + ICL | `--mode Meta`                 |