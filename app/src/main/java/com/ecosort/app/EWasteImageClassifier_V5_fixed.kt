package com.ecosort.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.json.JSONArray
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File

data class ClassificationResult(
    val category: String,
    val confidence: Float
)

class EWasteImageClassifier(context: Context) {

    private val appContext = context.applicationContext
    private var module: Module? = null
    private lateinit var labels: List<String>

    companion object {
        private const val IMAGE_SIZE = 224
    }

    // Load the V5 model and its class mapping from app/src/main/assets.
    fun loadModel() {
        if (module != null) return

        val modelFile = File(appContext.filesDir, "classifier.pte")

        if (!modelFile.exists()) {
            appContext.assets.open("classifier.pte").use { input ->
                modelFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }

        val labelsText = appContext.assets
            .open("labels.json")
            .bufferedReader()
            .use { it.readText() }

        // V5 labels.json is a JSON OBJECT: {"0":"Battery", ... "11":"Wires"}.
        // Also accept a JSON ARRAY in case the labels file is changed later.
        labels = when (val first = labelsText.trim().firstOrNull()) {
            '{' -> {
                val labelsObject = JSONObject(labelsText)
                List(labelsObject.length()) { index ->
                    labelsObject.getString(index.toString())
                }
            }
            '[' -> {
                val labelsArray = JSONArray(labelsText)
                List(labelsArray.length()) { index ->
                    labelsArray.getString(index)
                }
            }
            else -> throw IllegalStateException(
                "labels.json must contain a JSON object or array"
            )
        }

        if (labels.size != 12) {
            throw IllegalStateException(
                "V5 expects 12 class labels, but found ${labels.size}"
            )
        }

        module = Module.load(modelFile.absolutePath)
    }

    // V5 preprocessing: RGB, resize to 224x224, scale to [0,1], HWC to CHW.
    // No ImageNet mean/std normalization is applied.
    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmap,
            IMAGE_SIZE,
            IMAGE_SIZE,
            true
        )

        val inputData = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)

        for (y in 0 until IMAGE_SIZE) {
            for (x in 0 until IMAGE_SIZE) {
                val pixel = resizedBitmap.getPixel(x, y)
                val index = y * IMAGE_SIZE + x

                inputData[index] =
                    Color.red(pixel) / 255.0f

                inputData[IMAGE_SIZE * IMAGE_SIZE + index] =
                    Color.green(pixel) / 255.0f

                inputData[2 * IMAGE_SIZE * IMAGE_SIZE + index] =
                    Color.blue(pixel) / 255.0f
            }
        }

        if (resizedBitmap !== bitmap) {
            resizedBitmap.recycle()
        }

        return inputData
    }

    fun classify(bitmap: Bitmap): ClassificationResult {
        val loadedModule = module
            ?: throw IllegalStateException("Classifier model not loaded")

        if (!::labels.isInitialized) {
            throw IllegalStateException("Classifier labels not loaded")
        }

        val inputData = preprocessImage(bitmap)

        val inputTensor = Tensor.fromBlob(
            inputData,
            longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())
        )

        val outputTensor = loadedModule.forward(
            EValue.from(inputTensor)
        )[0].toTensor()

        val scores = outputTensor.dataAsFloatArray

        if (scores.size != labels.size) {
            throw IllegalStateException(
                "Model returned ${scores.size} scores, but labels.json contains ${labels.size} labels"
            )
        }

        // V5 documentation specifies softmax over the 12 output scores.
        val maxScore = scores.maxOrNull()
            ?: throw IllegalStateException("Model returned no class scores")

        val exponentials = scores.map {
            kotlin.math.exp((it - maxScore).toDouble())
        }
        val sum = exponentials.sum()

        if (!sum.isFinite() || sum <= 0.0) {
            throw IllegalStateException("Model returned invalid class scores")
        }

        val probabilities = exponentials.map { (it / sum).toFloat() }
        val bestIndex = probabilities.indices.maxByOrNull {
            probabilities[it]
        } ?: throw IllegalStateException("Could not select a predicted class")

        return ClassificationResult(
            category = labels[bestIndex],
            confidence = probabilities[bestIndex]
        )
    }

    fun unloadModel() {
        module?.close()
        module = null
    }
}
