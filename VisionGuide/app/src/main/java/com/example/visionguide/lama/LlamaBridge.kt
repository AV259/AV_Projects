package com.example.visionguide.lama

class LlamaBridge {
    companion object {
        init {
            // Loads native library: libllama-jni.so
            System.loadLibrary("llama-jni")
        }

        // Declare a native function implemented in C++
        external fun initLlamaModel(modelPath: String): Boolean
        external fun runInference(prompt: String, visualTokens: FloatArray): String
        external fun cleanup()

        // You can add more native functions, e.g.:
        // external fun runLlama(prompt: String, modelPath: String): String
    }
}