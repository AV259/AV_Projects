package com.example.visionguide

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream

class Projector(context: Context) {

    private val interpreter: Interpreter

    val batchSize = 1           // Always 1 for single image
    val clipSeqLength = 577   // CLIP ViT-L/14@336 outputs 577 tokens
    val outputSeq = 144       // projector downsamples to 144 tokens
    val embeddingDim = 1024   // embedding dimension

    init {
        val modelBuffer = loadModelFile(context, "ldpv2_projector_v2.tflite")
        interpreter = Interpreter(modelBuffer)
        Log.i("Projector", "Interpreter initialized successfully")
    }

    private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun runProjection(clipEmbedding: FloatArray): FloatArray {
        Log.i("Projector", "Received clipEmbedding size: ${clipEmbedding.size}")

        if (clipEmbedding.size != clipSeqLength * embeddingDim) {
            throw IllegalArgumentException(
                "clipEmbedding size mismatch: expected ${clipSeqLength * embeddingDim}, got ${clipEmbedding.size}"
            )
        }

        val inputArray = Array(1) { Array(clipSeqLength) { FloatArray(embeddingDim) } }
        var idx = 0
        for (i in 0 until clipSeqLength) {
            for (j in 0 until embeddingDim) {
                inputArray[0][i][j] = clipEmbedding[idx++]
            }
        }
        Log.i("Projector", "Input array shape: [${inputArray.size}, ${inputArray[0].size}, ${inputArray[0][0].size}]")

        // Output placeholder → [1, 144, 1024]
        val outputArray = Array(1) { Array(outputSeq) { FloatArray(embeddingDim) } }
        Log.i("Projector", "Output array shape: [${outputArray.size}, ${outputArray[0].size}, ${outputArray[0][0].size}]")

        try {
            interpreter.run(inputArray, outputArray)
        } catch (e: Exception) {
            Log.e("Projector", "Interpreter run failed: ${e.message}")
            throw e
        }

        // Flatten output to 1D for JNI or further processing
        val flattenedOutput = outputArray[0].flatMap { it.toList() }.toFloatArray()
        Log.i("Projector", "Flattened output size: ${flattenedOutput.size}")

        return flattenedOutput
    }
}