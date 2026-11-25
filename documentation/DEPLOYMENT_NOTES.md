# Deployment notes

## Included in the APK
- Android application source and resources
- MobileNetV3 classifier asset (`classifier.pte`)
- Classification labels
- E-waste knowledge-base JSON assets

## Required separately for Llama
The current code loads the Llama model and tokenizer from:

`<app-specific external files directory>/llama/`

Required filenames:
- `Llama-3.2-1B-Instruct-8da4w-gptq-2k.pte`
- `tokenizer.json`

The Llama model is approximately 915 MB and is intentionally excluded from this repository. The current snapshot does not implement automatic first-launch downloading or provisioning. Supply the files through a controlled evaluator download/offline package and test the exact setup on a clean phone.

## Before public submission
- Confirm permission/licensing for model weights and datasets.
- Confirm no secrets or personal files are present.
- Build and test a release APK on a clean device.
- Confirm both classification and Llama chat work after fresh installation.
