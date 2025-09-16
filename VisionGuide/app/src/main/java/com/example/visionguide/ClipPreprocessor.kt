package com.example.visionguide

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.*
import java.nio.FloatBuffer
import android.util.Log
import java.io.File

class ClipPreprocessor(private val context: Context) {

    companion object {
        private const val TARGET_SIZE = 336
        private const val CHANNELS = 3
    }

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null
    private var isLoaded = false

    // Load model in background
    fun loadModelAsync(onLoaded: (success: Boolean, error: String?) -> Unit) {
        Thread {
            try {
                Log.i("ClipPreprocessor", "Loading ONNX model in background thread")

                // Copy from assets if not already present
                val modelFile = File(context.filesDir, "clip_vitL14_336_encoder_qlinear.onnx")
                if (!modelFile.exists()) {
                    context.assets.open("clip_vitL14_336_encoder_qlinear.onnx").use { input ->
                        modelFile.outputStream().use { output -> input.copyTo(output) }
                    }
                    Log.i("ClipPreprocessor", "Model copied to ${modelFile.absolutePath}")
                }

                val sessionOptions = OrtSession.SessionOptions().apply {
                    setInterOpNumThreads(1)
                    setIntraOpNumThreads(1)
                    setMemoryPatternOptimization(false) // important for large models on Android
                }

                // Load model by file path, not bytes
                ortSession = ortEnv.createSession(modelFile.absolutePath, sessionOptions)
                isLoaded = true

                Log.i("ClipPreprocessor", "ONNX model loaded successfully")
                onLoaded(true, null)

            } catch (oom: OutOfMemoryError) {
                Log.e("ClipPreprocessor", "OOM while loading model", oom)
                onLoaded(false, "Device ran out of memory while loading model")
            } catch (e: Exception) {
                Log.e("ClipPreprocessor", "Failed to load ONNX model", e)
                onLoaded(false, e.message)
            }
        }.start()
    }

    private fun checkModelReady() {
        if (!isLoaded || ortSession == null) {
            throw IllegalStateException("ONNX model not loaded yet")
        }
    }

    /**
     * Resize bitmap to 336x336
     */
    private fun resize(bitmap: Bitmap): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, TARGET_SIZE, TARGET_SIZE, true)
    }

    /**
     * Convert bitmap to normalized FloatArray
     */
    private fun bitmapToFloatArray(bitmap: Bitmap): FloatArray {
        val resized = resize(bitmap)
        val inputSize = TARGET_SIZE * TARGET_SIZE
        val floatValues = FloatArray(CHANNELS * inputSize)
        val pixels = IntArray(inputSize)
        resized.getPixels(pixels, 0, TARGET_SIZE, 0, 0, TARGET_SIZE, TARGET_SIZE)

        for (i in 0 until inputSize) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            floatValues[i] = (r - 0.5f) / 0.5f
            floatValues[i + inputSize] = (g - 0.5f) / 0.5f
            floatValues[i + 2 * inputSize] = (b - 0.5f) / 0.5f
        }

        return floatValues
    }

    /**
     * Run inference and return CLIP image embedding
     * @return a float array
     */
    fun encodeImage(bitmap: Bitmap, projector: Projector): FloatArray {
        checkModelReady()  // Ensure model is loaded
        val inputArray = bitmapToFloatArray(bitmap)
        val inputBuffer = FloatBuffer.wrap(inputArray)
        val shape = longArrayOf(1, 3, TARGET_SIZE.toLong(), TARGET_SIZE.toLong())

        val tensor = OnnxTensor.createTensor(ortEnv, inputBuffer, shape)

        val session = ortSession ?: throw IllegalStateException("ONNX session not initialized")
        val inputName = session.inputNames.iterator().next()

        session.run(mapOf(inputName to tensor)).use { results ->
            val rawOutput = results[0].value
            when (rawOutput) {
                is Array<*> -> {
                    // rawOutput: float[1][N][D]
                    val batch = rawOutput[0] as Array<FloatArray>
                    // Flatten into 1D for JNI
                    val embeddingList: List<Float> = batch.flatMap { it.asList() }  // convert each FloatArray to List<Float>
                    val embedding: FloatArray = embeddingList.toFloatArray()        // convert to FloatArray

                    if (embedding.size % projector.embeddingDim != 0) {
                        throw IllegalStateException("Embedding size ${embedding.size} not divisible by expected dim ${projector.embeddingDim}")
                    }
                    return embedding
                }
                else -> throw IllegalStateException("Unexpected ONNX output type: ${rawOutput?.javaClass}")
            }
        }
    }
}
