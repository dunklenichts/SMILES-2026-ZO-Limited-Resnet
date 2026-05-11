## Reproducibility
Run the solution from the repository root:

```bash
pip install -r requirements.txt
python validate.py --data_dir ./data --batch_size 32 --n_batches 1 --seed 42 --output results.json
```
This command uses the official validate.py script. I did not modify validate.py or model.py

Final result:
```json
{
  "val_accuracy_top1_imagenet_head": 0.0037,
  "val_accuracy_top1_init_head": 0.2046,
  "val_accuracy_top1_finetuned": 0.2063,
  "n_batches": 1,
  "batch_size": 32,
  "layers_tuned": [
    "fc.weight",
    "fc.bias"
  ],
  "total_samples": 10000
}
```
_____

## Modified files
* head_init.py
* zo_optimizer.py
* augmentation.py
The main improvement is in head_init.py

## First solution (non-semantic)
Before the final approach, I tried several solutions without using semantic ImageNet-to-CIFAR100 initialization.

First, I used a small-scale Xavier initialization for the new CIFAR100 head:
```python
nn.init.xavier_uniform_(layer.weight)
layer.weight.data.mul_(0.01)
nn.init.zeros_(layer.bias)
```
Then I applied zero-order fine-tuning to ```["fc.weight", "fc.bias"]``` and the reproduced result for this version was:
```json
{
  "val_accuracy_top1_imagenet_head": 0.0037,
  "val_accuracy_top1_init_head": 0.0121,
  "val_accuracy_top1_finetuned": 0.0149,
  "n_batches": 512,
  "batch_size": 16,
  "layers_tuned": [
    "fc.weight",
    "fc.bias"
  ],
  "total_samples": 10000
}
```
This improved the initialized head only slightly, from about 1.2% to about 1.49%. I also tested different compute-budget splits: more steps with smaller batches and fewer steps with larger batches, but this did not give a strong improvement.

Then I tried different layer selection strategies. I optimized only `fc.bias`, only `fc.weight`, and both `fc.weight` and `fc.bias`. Optimizing only one part of the head did not improve the result. The best non-semantic variant was still tuning both `fc.weight` and `fc.bias`, but the accuracy stayed close to 1.49%.

I also tested several standard initialization methods for the final head: Xavier without small scaling, normal initialization, orthogonal initialization, and small-scale Xavier initialization. Small-scale Xavier was the most stable among these options, but it was still not enough because the classifier head started almost randomly.

For the optimizer, I replaced the naive finite-difference estimator with an SPSA-style estimator using random -1/+1 perturbations. I also added an Adam-style moment update for the zero-order pseudo-gradients. This made the optimizer more stable, but it still could not train the randomly initialized 100-class head well within the limited compute budget.

I also tried stronger augmentations, but they decreased validation accuracy. Most likely, they made the zero-order loss estimates noisier.

After these experiments, I concluded that the main problem was not the number of zero-order steps, but the random initialization of the CIFAR100 head. So, to improve the starting point, I used semantic initialization for the final CIFAR100 head. Instead of initializing all 100 classes only with random weights, I used the pretrained ImageNet classifier head from ResNet18.

The earlier non-semantic experiments are saved in a separate branch first-solution

To reproduce the best non-semantic baseline, run:
```python
git checkout first-solution
python validate.py --data_dir ./data --batch_size 16 --n_batches 512 --seed 42 --output results_old.json
```

### Experiments before semantic initialization
| Experiment | Result / reason not used |
|---|---|
| Small-scale Xavier initialization + ZO fine-tuning | `0.0149` only small improvement over `0.0121` initialized head |
| Xavier without small scale | `0.0120` worse than small-scale Xavier |
| Normal initialization (`std=0.01`) | `0.0088` worse result |
| Orthogonal initialization with small scale | `0.0094` worse result |
| Fine-tuning only `fc.weight` before semantic init | `0.0120` did not help |
| Fine-tuning only `fc.bias` before semantic init | `0.0074` degraded the result |
| SPSA with `fc.weight + fc.bias`, `128 × 32` before semantic init | `0.0141` still below the old best `0.0149` |
| Stronger augmentation | `0.0111` decreased validation accuracy |


## Final approach
The initialized model gives almost random accuracy because of a random 100-class head. It is also hard for a zero-order optimizer to train this head from scratch, because the optimizer only receives scalar loss values and does not use backpropagation.

To improve the starting point, I used semantic initialization for the final CIFAR100 head. Instead of initializing all 100 classes only with random weights, I used the pretrained ImageNet classifier head from ResNet18. The method works as follows:
1. Load CIFAR100 class names from the local CIFAR100 metadata file.
2. Load ImageNet class names and classifier weights from pretrained ResNet18.
3. For each CIFAR100 class, search for related ImageNet class names.
4. If related ImageNet classes are found, initialize the CIFAR100 classifier row with the average ImageNet classifier weight.
5. If no related ImageNet class is found, then use small-scale Xavier initialization as a fallback.
This improved the initialized-head accuracy from about 0.012 to about 0.20

**IMPORTANT**: This does not use validation labels. It only uses public CIFAR100 class names, public ImageNet class names, and the pretrained ImageNet classifier head.

### Experiments after semantic initialization
| Experiment | Result / reason not used |
|---|---|
| ImageNet bias transfer after semantic init | `0.2045` initialized head, slightly worse than `0.2046` without bias |
| Different semantic scales | around `0.2046` initialized head, almost no effect |
| Fine-tuning only `fc.weight` after semantic init | `0.2063` close, but not better than final |
| Fine-tuning only `fc.bias` after semantic init | `0.2048` did not help |
| Longer fine-tuning after semantic init | `0.2062` not better than short fine-tuning |
| Final semantic initialization + short ZO fine-tuning | `0.2063` selected final result |
