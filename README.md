# EcoSort AI

An Android application for e-waste identification and responsible handling guidance.

EcoSort combines:
- **MobileNetV3-Small (ExecuTorch)** for image-based classification across 12 e-waste categories.
- **Llama 3.2 1B (ExecuTorch)** for conversational explanations and safety/disposal guidance.
- A local JSON knowledge base containing category-specific information.

## Supported classification labels

Battery, Bulb, IC, Keyboard, Laptop, Mouse, Passive Components, PCB, Phone, Resistor, Transistor, and Wires.

## Repository contents

- `app/` — Android application source, resources, model assets, and knowledge-base JSON files.
- `gradle/`, Gradle wrapper files, and Gradle build scripts — project build configuration.

## Build the Android app

1. Install Android Studio and a compatible Android SDK.
2. Clone this repository and open the project root in Android Studio.
3. Allow Gradle sync to complete.
4. Build and run the `app` configuration.

The project uses the Android Gradle Plugin and ExecuTorch Android runtime. Dependency versions are defined in `gradle/libs.versions.toml` and `app/build.gradle.kts`.

## Model files

The MobileNet classifier package (`app/src/main/assets/classifier.pte`) and `labels.json` are included in this repository.

The Llama model is **not included** in this repository. The current app code expects these files in its app-specific external files directory under `llama/`:

- `Llama-3.2-1B-Instruct-8da4w-gptq-2k.pte`
- `tokenizer.json`

Create the `llama` directory in the app-specific external files directory and place both files there before using the Llama-powered chat feature. The model file is large (approximately 915 MB); do not commit it to GitHub. For evaluator distribution, provide the model through an approved download or offline package and document the exact setup steps.

## Current deployment notes

- The app's image-classification model is bundled in the APK as an asset.
- Llama model provisioning is currently a separate setup step; first-launch automatic download/provisioning is not yet implemented.
- Test on a clean Android device before distributing a release APK.
- Do not claim full offline first-run setup until the Llama model and tokenizer are provisioned on the device.

## Privacy and security

Do not commit `local.properties`, signing keys, passwords, API keys, personal data, or large generated build/cache directories. Review all files before making the repository public.

## Project status

This repository reflects the Android project snapshot supplied for submission preparation. Validate the app on a clean device and update screenshots, test results, and deployment instructions before final submission.
