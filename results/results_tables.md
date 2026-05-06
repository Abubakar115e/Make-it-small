# Results tables

## Baseline

| Metric | Value |
|---|---:|
| Dataset | CIFAR-100 |
| Model | ResNet-18 |
| Accuracy | 81.89% |
| Dense size | 42.83 MiB |

## Fine-grained pruning

| Target sparsity | Accuracy after pruning | Accuracy after fine-tuning | Approx. non-zero model size |
|---:|---:|---:|---:|
| 50% | 77.30% | 81.64% | 21.43 MiB |
| 70% | 29.47% | 80.65% | 12.88 MiB |
| 90% | 1.23% | 58.97% | 4.32 MiB |
| 95% | 1.00% | 19.17% | 2.18 MiB |

## Channel pruning

| Prune ratio | Accuracy after pruning | Accuracy after fine-tuning | MAC reduction |
|---:|---:|---:|---:|
| 0.01 | 81.84% | 81.80% | 1.05% |
| 0.05 | 81.27% | 81.70% | 4.50% |
| 0.10 | 80.16% | 81.85% | 9.12% |
| 0.20 | 75.29% | 81.26% | 18.56% |
| 0.30 | 68.07% | 80.81% | 27.55% |
| 0.40 | 50.90% | 80.08% | 36.98% |
| 0.50 | 25.66% | 79.09% | 46.11% |

## K-means quantization with QAT

| Bitwidth | Accuracy before QAT | Accuracy after QAT | Size reduction |
|---:|---:|---:|---:|
| 8 | 80.59% | 81.73% | 4.00x |
| 6 | 74.10% | 81.37% | 5.33x |
| 4 | 59.13% | 80.92% | 8.00x |
| 3 | 5.49% | 78.70% | 10.67x |
