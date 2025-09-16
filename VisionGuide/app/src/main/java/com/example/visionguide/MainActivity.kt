package com.example.visionguide

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.speech.RecognizerIntent
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.visionguide.lama.LlamaBridge
import java.io.File
import java.util.*
import  java.io.IOException
import android.widget.Toast
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var resultText: TextView
    private lateinit var imageView: ImageView
    private lateinit var capturedBitmap: Bitmap
    private var spokenText: String = ""
    private lateinit var tts: TextToSpeech
    private var isTTSInitialized = false
    private lateinit var clipPreprocessor: ClipPreprocessor;
    private lateinit var projector: Projector
    private var isLlamaInitialized: Boolean = false
    private var input = 0;
    private lateinit var speakButton: Button
    private lateinit var photoButton: Button
    private var isImageCaptured = false
    private var isSpeechCaptured = false



    companion object {
        private const val CAMERA_PERMISSION_CODE = 100
        private const val AUDIO_PERMISSION_CODE = 101
        private const val TAG = "MainActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            setContentView(R.layout.activity_main)

            val assetFileName = "mobilellama-1.4b-q4_0.gguf"
            val modelFile = File(filesDir, assetFileName)

            // Copy model and initialize LLaMA
            copyModelFromAssets(assetFileName, modelFile)
            initializeLlama()

            // Initialize UI components with error handling
            initializeUI()

            // Initialize other components
            initializeTTS()
            initializeMLComponents()

        } catch (e: Exception) {
            handleError("Failed to initialize application", e.message ?: "Unknown error", e)
            // Show user-friendly error and potentially close app
            Toast.makeText(this, "App failed to start. Please restart.", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    private fun copyModelFromAssets(assetName: String, targetFile: File) {
        try {
            assets.open(assetName).use { input ->
                FileOutputStream(targetFile).use { output ->
                    val buffer = ByteArray(1024 * 1024) // 1 MB buffer
                    var totalBytes: Long = 0
                    var bytesRead: Int
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                        totalBytes += bytesRead
                    }
                    output.flush()
                    Log.i(TAG, "Model copied successfully: ${targetFile.absolutePath}, size=$totalBytes")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error copying model from assets", e)
        }
    }

    private fun initializeLlama() {
        val modelFile = File(filesDir, "mobilellama-1.4b-q4_0.gguf")

        // Check if model file exists and is valid
        if (!modelFile.exists() || modelFile.length() < 100_000_000L) { // sanity check ~100MB
            Log.e(TAG, "Model file missing or invalid at: ${modelFile.absolutePath}")
            speakText("Language model not found, please check installation")
            return
        }

        Thread {
            try {
                val modelPath = modelFile.absolutePath
                Log.i(TAG, "Initializing LLaMA with path: $modelPath, exists=${modelFile.exists()}, size=${modelFile.length()}")

                val success = LlamaBridge.initLlamaModel(modelPath) // should return Boolean
                runOnUiThread {
                    if (success) {
                        isLlamaInitialized = true
                        speakText("Language model loaded successfully")
                        speakButton.isEnabled = true
                        photoButton.isEnabled = true
                    } else {
                        isLlamaInitialized = false
                        speakText("Failed to load language model")
                        handleError("Failed to load language model", "JNI returned false")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize LLaMA model", e)
                runOnUiThread {
                    isLlamaInitialized = false
                    handleError("Failed to load language model", e.message ?: "Unknown error", e)
                    speakText("Failed to initialize language model")
                }
            }
        }.start()
    }


    @Suppress("DEPRECATION")
    private val captureImageLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val data = result.data
        if (result.resultCode == RESULT_OK && data != null) {
            val photo = data.extras?.get("data") as? Bitmap
            if (photo != null) {
                markImageCaptured(photo)
            } else {
                handleError("Failed to capture image", "Image data is null")
            }
            checkForBothInputs();
        }  else {
            handleError("Camera capture failed", "Result code: ${result.resultCode}")
        }
    }

    @SuppressLint("SetTextI18n")
    private val speechLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        try {
            if (result.resultCode == RESULT_OK && result.data != null) {
                val spoken = result.data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
                spokenText = spoken?.firstOrNull() ?: ""
                if (spokenText.isNotEmpty()) {
                    markSpeechCaptured(spokenText)
                } else {
                    handleError("Speech recognition failed", "No speech detected")
                }
            } else {
                handleError("Speech recognition cancelled or failed", "Result code: ${result.resultCode}")
            }
        } catch (e: Exception) {
            handleError("Error processing speech input", e.message ?: "Unknown error", e)
        }
    }

    private fun resetInputs() {
        isImageCaptured = false
        isSpeechCaptured = false
        capturedBitmap = Bitmap.createBitmap(1,1, Bitmap.Config.ARGB_8888) // placeholder reset
        spokenText = ""
    }

    @SuppressLint("SetTextI18n")
    private fun markImageCaptured(bitmap: Bitmap) {
        capturedBitmap = bitmap
        isImageCaptured = true
        imageView.setImageBitmap(bitmap)
        resultText.text = "Image captured ✅"
        speakText("Image captured successfully")
        checkForBothInputs()
    }

    @SuppressLint("SetTextI18n")
    private fun markSpeechCaptured(text: String) {
        spokenText = text
        isSpeechCaptured = true
        resultText.text = "You said: $spokenText"
        speakText("Voice input received")
        checkForBothInputs()
    }

    private fun initializeUI() {
        try {
            resultText = findViewById(R.id.labelText)
            imageView = findViewById(R.id.imageView)
            speakButton = findViewById(R.id.audioTab)
            photoButton = findViewById(R.id.imageButton)

            // Disable buttons until LLaMA is initialized
            speakButton.isEnabled = false
            photoButton.isEnabled = false

            // speech
            speakButton.setOnClickListener {
                try {
//                    speakText("Please speak your instruction")
                    requestPermission(Manifest.permission.RECORD_AUDIO, AUDIO_PERMISSION_CODE) {
                        startSpeechInput()
                    }
                } catch (e: Exception) {
                    handleError("Error starting speech input", e.message ?: "Unknown error", e)
                }
            }

            //image
            photoButton.setOnClickListener {
                try {
                    speakText("Opening camera to capture image")
                    requestPermission(Manifest.permission.CAMERA, CAMERA_PERMISSION_CODE) {
                        openCamera()
                    }
                } catch (e: Exception) {
                    handleError("Error starting camera", e.message ?: "Unknown error", e)
                }
            }
        } catch (e: Exception) {
            throw Exception("Failed to initialize UI components", e)
        }
    }

    private fun initializeTTS() {
        try {
            tts = TextToSpeech(this) { status ->
                if (status == TextToSpeech.SUCCESS) {
                    val result = tts.setLanguage(Locale.US)
                    if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                        Log.w(TAG, "TTS language not supported")
                        // Try default language
                        tts.setLanguage(Locale.getDefault())
                    }
                    isTTSInitialized = true
                    Log.i(TAG, "TTS initialized successfully")
                    speakText("TTS loaded successfully")
                } else {
                    Log.e(TAG, "TTS initialization failed with status: $status")
                    handleError("Text-to-speech initialization failed", "Status code: $status")
                }
            }
        } catch (e: Exception) {
            handleError("Error initializing text-to-speech", e.message ?: "Unknown error", e)
        }
    }

    private fun initializeMLComponents() {
        clipPreprocessor = ClipPreprocessor(this)

        // Load ONNX model in background
        clipPreprocessor.loadModelAsync { success, error ->
            runOnUiThread {
                if (success) {
                    Log.i(TAG, "ONNX model loaded successfully")
                    speakText("ONNX model loaded successfully")
                    projector = Projector(this)
                } else {
                    Log.e(TAG, "Failed to load ONNX model: $error")
                    handleError("Failed to load ML model", error ?: "Unknown error")
                }
            }
        }
    }


    private fun checkForBothInputs() {
        try {
            if (!isImageCaptured || !isSpeechCaptured) return  // wait until both are ready

            if (!isLlamaInitialized) {
                handleError("Cannot process request", "Language model not initialized")
                speakText("Please wait for the model to finish loading")
                return
            }

            val hv = preprocessInput()
            speakText("Processing your request...")

            if (spokenText.isEmpty()) {
                Log.e("LlamaBridge", "Prompt is empty, skipping inference")
                return
            }
            if (spokenText.length > 4096) { // adjust based on model’s context length
                Log.e("LlamaBridge", "Prompt too long: ${spokenText.length}")
                return
            }

            // LLaMA inference
            Thread {
                try {
                    Log.i(TAG, "Starting LLaMA inference...")
                    val output = LlamaBridge.runInference(spokenText, hv)

                    if (output.isEmpty()) {
                        throw Exception("Model returned empty response")
                    }

                    runOnUiThread {
                        resultText.text = output
                        speakText(output)
                        resetInputs()  // reset for next request
                    }
                    Log.i(TAG, "LLaMA inference completed successfully")

                } catch (e: Exception) {
                    Log.e(TAG, "LLaMA inference failed", e)
                    runOnUiThread {
                        handleError("Error processing your request", e.message ?: "Unknown error", e)
                        speakText("I'm sorry, I couldn't process your request. Please try again.")
                        resetInputs()  // reset flags to allow retry
                    }
                }
            }.start()

        } catch (e: Exception) {
            handleError("Error checking inputs", e.message ?: "Unknown error", e)
        }
    }

    private fun preprocessInput(): FloatArray {
        try {
            Log.i(TAG, "Starting image preprocessing...")
            val fvArray = clipPreprocessor.encodeImage(capturedBitmap, projector)

            if (fvArray.isEmpty()) {
                throw Exception("CLIP preprocessing returned empty array")
            }

            Log.i(TAG, "Running projection...")
            val hv = projector.runProjection(fvArray)

            if (hv.isEmpty()) {
                throw Exception("Projector returned empty array")
            }

            Log.i(TAG, "Preprocessing completed successfully")
            return hv

        } catch (e: Exception) {
            Log.e(TAG, "Preprocessing failed", e)
            throw Exception("Failed to preprocess image: ${e.message}", e)
        }
    }


    private fun speakText(message: String) {
        try {
            if (isTTSInitialized && ::tts.isInitialized) {
                val result = tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
                if (result == TextToSpeech.ERROR) {
                    Log.w(TAG, "TTS speak failed for message: $message")
                }
            } else {
                Log.w(TAG, "TTS not initialized, cannot speak: $message")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in speakText", e)
            // Don't throw here as TTS is not critical for core functionality
        }
    }


    private fun requestPermission(permission: String, requestCode: Int, onGranted: () -> Unit) {
        try {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(permission), requestCode)
            } else {
                onGranted()
            }
        } catch (e: Exception) {
            handleError("Error requesting permission", e.message ?: "Unknown error", e)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        try {
            when (requestCode) {
                CAMERA_PERMISSION_CODE -> {
                    if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                        openCamera()
                    } else {
                        handleError("Camera permission denied", "Cannot capture images without camera permission")
                        speakText("Camera permission denied")
                    }
                }
                AUDIO_PERMISSION_CODE -> {
                    if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                        startSpeechInput()
                    } else {
                        handleError("Audio permission denied", "Cannot record speech without microphone permission")
                        speakText("Audio permission denied")
                    }
                }
            }
        } catch (e: Exception) {
            handleError("Error handling permission result", e.message ?: "Unknown error", e)
        }
    }


    private fun openCamera() {
        try {
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            if (cameraIntent.resolveActivity(packageManager) != null) {
                captureImageLauncher.launch(cameraIntent)
            } else {
                handleError("Camera not available", "No camera app found on device")
                speakText("Camera not available on this device")
            }
        } catch (e: Exception) {
            handleError("Error opening camera", e.message ?: "Unknown error", e)
        }
    }

    private fun startSpeechInput() {
        try {
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
                putExtra(RecognizerIntent.EXTRA_PROMPT, "Speak now...")
                putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            }

            if (intent.resolveActivity(packageManager) != null) {
                speechLauncher.launch(intent)
            } else {
                handleError("Speech recognition not available", "No speech recognition app found")
                speakText("Speech recognition not available on this device")
            }
        } catch (e: Exception) {
            handleError("Error starting speech input", e.message ?: "Unknown error", e)
        }
    }

    private fun handleError(userMessage: String, technicalMessage: String, exception: Exception? = null) {
        Log.e(TAG, "$userMessage: $technicalMessage", exception)

        // Show user-friendly message
        runOnUiThread {
            Toast.makeText(this, userMessage, Toast.LENGTH_SHORT).show()
            resultText.text = "Error: $userMessage $technicalMessage"
        }

        // Optional: Report to crash analytics (Firebase Crashlytics, etc.)
        // CrashlyticsHelper.recordError(exception ?: Exception(technicalMessage))
    }

    override fun onDestroy() {
        try {
            // Clean up TTS first
            if (::tts.isInitialized) {
                tts.stop()
                tts.shutdown()
            }

            // Clean up LLaMA resources
            cleanupLlama()

        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup", e)
        } finally {
            super.onDestroy()
        }
    }

    // Called when app is moved to background or killed
    override fun onStop() {
        super.onStop()
        super.onLowMemory()
        try {
            Log.w(TAG, "Low memory warning - cleaning up resources")
//            cleanupLlama()
        } catch (e: Exception) {
            Log.e(TAG, "Error during low memory cleanup", e)
        }
    }

    // Called when device is low on memory
    override fun onLowMemory() {
        try {
            super.onLowMemory()
            // Cleanup to free memory
//            cleanupLlama()
        } catch (e: Exception) {
            Log.e(TAG, "Error during memory trim cleanup", e)
        }
    }

    private fun cleanupLlama() {
        if (isLlamaInitialized) {
            Thread {
                try {
                    Log.i(TAG, "Cleaning up LLaMA resources...")
                    LlamaBridge.cleanup()
                    isLlamaInitialized = false
                    Log.i(TAG, "LLaMA cleanup completed")
                } catch (e: Exception) {
                    Log.e(TAG, "Error during LLaMA cleanup", e)
                }
            }.start()
        }
    }
}
