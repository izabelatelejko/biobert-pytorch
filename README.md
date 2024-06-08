# BioBERT-PyTorch
This repository provides the PyTorch implementation of [BioBERT](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506).
You can easily use BioBERT with [transformers](https://github.com/huggingface/transformers).
This project is supported by the members of [DMIS-Lab](https://dmis.korea.ac.kr/) @ Korea University including Jinhyuk Lee, Wonjin Yoon, Minbyul Jeong, Mujeen Sung, and Gangwoo Kim.

## Setup on WSL

1. Install [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-downloads?target_os=Windows) (choose OS Windows)
2. Create venv with python=3.8 and activate
3. Install torch
```{sh}
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
4. Install requirements.txt

## Installation
```bash
# Install huggingface transformers
pip install transformers==3.0.0

# Download all datasets including NER/RE/QA
./download.sh
```
Note that you should also install `torch` (see [download instruction](https://pytorch.org/)) to use `transformers`.
If the download script does not work, you can manually download the datasets [here](https://drive.google.com/file/d/1cGqvAm9IZ_86C4Mj7Zf-w9CFilYVDl8j/view?usp=sharing) which should be unzipped in the current directory (`tar -xzvf datasets.tar.gz`).

## Models
We provide following versions of BioBERT in PyTorch (click [here](https://huggingface.co/dmis-lab) to see all).
You can use BioBERT in `transformers` by setting `--model_name_or_path` as one of them (see example below).
* `dmis-lab/biobert-base-cased-v1.2`: Trained in the same way as BioBERT-Base v1.1 but includes LM head, which can be useful for probing
* `dmis-lab/biobert-base-cased-v1.1`: BioBERT-Base v1.1 (+ PubMed 1M)
* `dmis-lab/biobert-large-cased-v1.1`: BioBERT-Large v1.1 (+ PubMed 1M)
* `dmis-lab/biobert-base-cased-v1.1-mnli`: BioBERT-Base v1.1 pre-trained on MNLI
* `dmis-lab/biobert-base-cased-v1.1-squad`: BioBERT-Base v1.1 pre-trained on SQuAD
* `dmis-lab/biobert-base-cased-v1.2`: BioBERT-Base v1.2 (+ PubMed 1M + LM head)

For other versions of BioBERT or for Tensorflow, please see the [README](https://github.com/dmis-lab/biobert) in the original BioBERT repository.
You can convert any version of BioBERT into PyTorch with [this](https://github.com/huggingface/transformers/blob/v3.5.1/src/transformers/convert_bert_original_tf_checkpoint_to_pytorch.py).

## RMSProp
To use RMSPROP add the following class to the Transformers library under path transformers/optimization.py:
```python
class RMSprop(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
    ):
        """
        Base RMSprop optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-2):
                The learning rate.
            alpha (`float`, defaults to 0.99):
                The alpha value is the decay rate of the squared gradients of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 0.0):
                The weight decay value for the optimizer.
            momentum (`float`, defaults to 0):
                The momentum value speeds up the optimizer by taking bigger steps.
            centered (`bool`, defaults to `False`):
                Whether the gradients are normalized by the variance. If `True`, it can help training at the expense of additional compute.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            block_wise (`bool`, defaults to `True`):
                Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
        """
        if alpha == 0:
            raise NotImplementedError("RMSprop with alpha==0.0 is not supported!")
        if centered:
            raise NotImplementedError("Centered RMSprop is not supported!")
        super().__init__(
            "rmsprop",
            params,
            lr,
            (alpha, momentum),
            eps,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
        )
```


## Example
For instance, to train BioBERT on the NER dataset (NCBI-disease), run as:

```bash
# Pre-process NER datasets
cd named-entity-recognition
./preprocess.sh

# Choose dataset and run
export DATA_DIR=../datasets/NER
export ENTITY=NCBI-disease
python run_ner.py \
    --data_dir ${DATA_DIR}/${ENTITY} \
    --labels ${DATA_DIR}/${ENTITY}/labels.txt \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
    --output_dir output/${ENTITY} \
    --max_seq_length 128 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --save_steps 1000 \
    --seed 1 \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir
```

Please see each directory for different examples. Currently, we provide
* [embedding/](https://github.com/dmis-lab/biobert-pytorch/tree/master/embedding): BioBERT embedding.
* [named-entity-recognition/](https://github.com/dmis-lab/biobert-pytorch/tree/master/named-entity-recognition): NER using BioBERT.
* [question-answering/](https://github.com/dmis-lab/biobert-pytorch/tree/master/question-answering): QA using BioBERT.
* [relation-extraction/](https://github.com/dmis-lab/biobert-pytorch/tree/master/relation-extraction): RE using BioBERT.

Most examples are modifed from [examples](https://github.com/huggingface/transformers/tree/master/examples) in Hugging Face transformers.

## Citation
```bibtex
@article{lee2020biobert,
  title={BioBERT: a pre-trained biomedical language representation model for biomedical text mining},
  author={Lee, Jinhyuk and Yoon, Wonjin and Kim, Sungdong and Kim, Donghyeon and Kim, Sunkyu and So, Chan Ho and Kang, Jaewoo},
  journal={Bioinformatics},
  volume={36},
  number={4},
  pages={1234--1240},
  year={2020},
  publisher={Oxford University Press}
}
```

## License and Disclaimer
Please see the LICENSE file for details. Downloading data indicates your acceptance of our disclaimer.


## Contact
For help or issues using BioBERT-PyTorch, please create an issue.
