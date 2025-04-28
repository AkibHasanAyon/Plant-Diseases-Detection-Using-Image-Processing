package com.example.plant_disease_ver1

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ImageClassifier(context: Context) {
    private var interpreter: Interpreter? = null
    private val inputSize = 128
    private val modelFileName = "plant_disease_model.tflite"

    // Labels formatted as an array of three strings for each disease: [disease_name, cause, treatment]
    private val labels = arrayOf(
        arrayOf("আপেল স্ক্যাব", "আর্দ্র আবহাওয়া ও অতিরিক্ত পানির কারণে ছত্রাক জন্মায়।", "আক্রান্ত পাতা ছেঁটে ফেলুন এবং কপার ছত্রাকনাশক স্প্রে করুন। নিয়মিত ছাঁটাই ও পানি নিষ্কাশনের ব্যবস্থা নিন।"),
        arrayOf("আপেল ব্ল্যাক রট", "পচা ফল বা পুরনো ডালে ছত্রাক জন্ম নিয়ে ছড়ায়।", "পচা ফল ও ডাল কেটে ফেলুন এবং ছত্রাকনাশক ব্যবহার করুন। গাছের নিচে পড়ে থাকা ফল নিয়মিত পরিষ্কার করুন।"),
        arrayOf("সিডার আপেল রস্ট", "আপেল গাছের কাছে সিডার গাছ থাকলে ছত্রাক ছড়ায়।", "সিডার গাছ সরান এবং ছত্রাকনাশক ব্যবহার করুন। কাছাকাছি সিডার গাছ না রাখলে রোগ হবে না।"),
        arrayOf("আপেল___সুস্থ (গাছটি সুস্থ! 🌱)", "", "গাছকে সুস্থ রাখতে নিয়মিত পানি এবং আলো ও সার দেওয়া উচিত।"),
        arrayOf("ব্লুবেরি___সুস্থ (গাছটি সুস্থ! 🌱)", "", "গাছকে সুস্থ রাখতে নিয়মিত পানি এবং আলো ও সার দেওয়া উচিত।"),
        arrayOf("চেরি পাউডারি মিলডিউ", "শুষ্ক আবহাওয়ায় পাতায় ছত্রাকের স্তর জমে।", "সালফার স্প্রে দিন এবং বাতাস চলাচলের ব্যবস্থা রাখুন। গাছ খুব ঘন না হলে রোগ কম হয়।"),
        arrayOf("চেরি (টকসহ)___সুস্থ (গাছটি সুস্থ! 🌱)", "", "গাছকে সুস্থ রাখতে নিয়মিত পানি এবং আলো ও সার দেওয়া উচিত।"),
        arrayOf("ভুট্টা সারকোসপোরা পাতা দাগ", "গরম ও আর্দ্র আবহাওয়া।", "পাতা ছেঁটে ফেলুন এবং ছত্রাকনাশক স্প্রে করুন। সঠিক পানি সরবরাহ নিশ্চিত করুন।"),
        arrayOf("ভুট্টা কমন রস্ট", "আর্দ্র ও ঠাণ্ডা আবহাওয়ায় ছত্রাকের বীজ ছড়ায়।", "রোগমুক্ত জাত লাগান এবং ছত্রাকনাশক ব্যবহার করুন। জমিতে আগের রোগমুক্ত ফসল চাষে সাহায্য পাবে।"),
        arrayOf("ভুট্টা নর্দার্ন পাতা ব্লাইট", "আর্দ্র আবহাওয়া ও দীর্ঘ সময় বৃষ্টির কারণে ছত্রাক আক্রমণ।", "গাছের ঘনত্ব কমিয়ে দিন এবং রোগমুক্ত জাত ব্যবহার করুন।"),
        arrayOf("ভুট্টা___সুস্থ (গাছটি সুস্থ! 🌱)", "", "গাছকে সুস্থ রাখতে নিয়মিত পানি এবং আলো ও সার দেওয়া উচিত।"),
        arrayOf("আঙ্গুর ব্ল্যাক রট", "আর্দ্র আবহাওয়া ও পচা ফল থেকে ছত্রাক ছড়ায়।", "আক্রান্ত ফল ও ডাল কেটে ফেলুন এবং ছত্রাকনাশক ব্যবহার করুন।"),
        arrayOf("আঙ্গুর এসকা (ব্ল্যাক মিজলস)", "অতিরিক্ত আর্দ্রতা ও দীর্ঘকালীন উচ্চ তাপমাত্রা।", "আক্রান্ত লতা সরিয়ে ফেলুন এবং ছত্রাকনাশক স্প্রে করুন।"),
        arrayOf("আঙ্গুর পাতা ব্লাইট (আইসারিওপসিস পাতা দাগ)", "বৃষ্টি ও আর্দ্রতার কারণে ছত্রাক ছড়ায়।", "গাছের ডাল ছেঁটে ফেলুন এবং ছত্রাকনাশক স্প্রে করুন।"),
        arrayOf("আঙ্গুর___সুস্থ --- (গাছটি সুস্থ! 🌱)", "", "গাছকে সুস্থ রাখতে নিয়মিত পানি আলো ও সার দেওয়া উচিত।"),
        arrayOf("কমলা হুয়াংলংবিং (সাইট্রাস গ্রিনিং)", "সাদা মাছি দ্বারা ভাইরাস সংক্রমণ।", "সাদা মাছি নিয়ন্ত্রণ করুন এবং আক্রান্ত গাছ সরান।"),
        arrayOf("পিচ ব্যাকটেরিয়াল স্পট", "আর্দ্র আবহাওয়া ও উচ্চ তাপমাত্রা।", "তামা ভিত্তিক ছত্রাকনাশক ব্যবহার করুন। আক্রান্ত গাছ সরান।"),
        arrayOf("পিচ___সুস্থ (গাছটি সুস্থ! 🌱)", "", "গাছকে সুস্থ রাখতে নিয়মিত পানি এবং আলো ও সার দেওয়া উচিত।"),
        arrayOf("বেল মরিচ ব্যাকটেরিয়াল স্পট", "শীতল ও আর্দ্র আবহাওয়া।", "বেল মরিচের ডাল ও পাতা ছেঁটে ফেলুন এবং কপার স্প্রে ব্যবহার করুন।"),
        arrayOf("বেল মরিচ___সুস্থ (গাছটি সুস্থ! 🌱)", "", "গাছকে সুস্থ রাখতে নিয়মিত পানি আলো ও সার দেওয়া উচিত।"),
        arrayOf("আলু আর্লি ব্লাইট", "শীতল ও আর্দ্র আবহাওয়া।", "অস্তিত্বশীল আলু জাত ব্যবহার করুন এবং সঠিক সার ও পানি ব্যবহার করুন।"),
        arrayOf("আলু লেট ব্লাইট", "ঠাণ্ডা ও আর্দ্র পরিবেশে ছত্রাকের বৃদ্ধি।", "ছত্রাকনাশক স্প্রে করুন এবং জমিতে আগের রোগমুক্ত ফসল চাষ করুন।"),
        arrayOf("আলু___সুস্থ (গাছটি সুস্থ! 🌱)", "", "গাছকে সুস্থ রাখতে নিয়মিত পানি আলো ও সার দেওয়া উচিত।"),
        arrayOf("রাস্পবেরি___সুস্থ (গাছটি সুস্থ! 🌱)", "", "গাছকে সুস্থ রাখতে নিয়মিত পানি আলো ও সার দেওয়া উচিত।"),
        arrayOf("সয়াবিন___সুস্থ (গাছটি সুস্থ! 🌱)", "", "গাছকে সুস্থ রাখতে নিয়মিত পানি আলো ও সার দেওয়া উচিত।"),
        arrayOf("স্কোয়াশ পাউডারি মিলডিউ", "আর্দ্র আবহাওয়া ও কম বাতাস চলাচল।", "সালফার স্প্রে দিন বাতাস চলাচলের ব্যবস্থা রাখুন।"),
        arrayOf("স্ট্রবেরি পাতার স্কর্চ", "গরম ও আর্দ্র আবহাওয়া।", "পাতা ছেঁটে ফেলুন এবং ছত্রাকনাশক স্প্রে করুন।"),
        arrayOf("স্ট্রবেরি___সুস্থ (গাছটি সুস্থ! 🌱)", "", "গাছকে সুস্থ রাখতে নিয়মিত পানি আলো ও সার দেওয়া উচিত।"),
        arrayOf("মিষ্টি আলু সুস্থ (গাছটি সুস্থ! 🌱)", "---", "গাছকে সুস্থ রাখতে নিয়মিত পানি আলো ও সার দেওয়া উচিত।"),
        arrayOf("টমেটো ব্যাকটেরিয়াল স্পট", "অতিরিক্ত আর্দ্রতা ও সঠিক পরিচর্যার অভাব।", "কপার ছত্রাকনাশক স্প্রে করুন।"),
        arrayOf("মিষ্টি আলু সুস্থ (গাছটি সুস্থ! 🌱)", "---", "গাছকে সুস্থ রাখতে নিয়মিত পানি আলো ও সার দেওয়া উচিত।"),
        arrayOf("টমেটো লেট ব্লাইট", "গরম ও আর্দ্র আবহাওয়া।", "রোগমুক্ত জাত ব্যবহার করুন এবং ছত্রাকনাশক স্প্রে করুন।"),
        arrayOf("টমেটো পাতার ছাঁচ", "ঠাণ্ডা ও আর্দ্র আবহাওয়া।", "আক্রান্ত অংশ ছেঁটে ফেলুন এবং ছত্রাকনাশক প্রয়োগ করুন।"),
        arrayOf("টমেটো সেপ্টোরিয়া দাগ", "আর্দ্র আবহাওয়া।", "আক্রান্ত অংশ কেটে ফেলুন এবং ছত্রাকনাশক ব্যবহার করুন।"),
        arrayOf("টমেটো স্পাইডার মাইট", "গরম ও শুকনো আবহাওয়া।", "নিম তেল স্প্রে করুন এবং পোকা নিয়ন্ত্রণ করুন।"),
        arrayOf("টমেটো টার্গেট স্পট", "গরম ও আর্দ্র আবহাওয়া।", "ছত্রাকনাশক স্প্রে এবং রোগমুক্ত জাত ব্যবহার করুন।"),
        arrayOf("টমেটো ইয়েলো লিফ কার্ল ভাইরাস", "সাদা মাছি দ্বারা ভাইরাস ছড়ায়।", "সাদা মাছি নিয়ন্ত্রণ করুন এবং আক্রান্ত গাছ সরান।"),
        arrayOf("টমেটো মোজাইক ভাইরাস", "ভাইরাস দ্বারা সংক্রমণ।", "রোগমুক্ত বীজ ব্যবহার করুন এবং গাছ ও পোকা নিয়ন্ত্রণ করুন।"),
        arrayOf("টমেটো___সুস্থ (গাছটি সুস্থ! 🌱)", "", "গাছকে সুস্থ রাখতে নিয়মিত পানি আলো ও সার দেওয়া উচিত।")
    )


    init {
        interpreter = Interpreter(loadModelFile(context))
        Log.d("ImageClassifier", "Interpreter initialized successfully.")
    }

    // Load the TFLite model from the assets folder.
    private fun loadModelFile(context: Context): MappedByteBuffer {
        Log.d("ImageClassifier", "Loading model file: $modelFileName")
        val fileDescriptor = context.assets.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // Preprocess the bitmap, run inference, and return the predicted label.
    fun classify(bitmap: Bitmap): String {
        // Resize the image to match the model's expected input size.
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        Log.d("ImageClassifier", "Resized bitmap to $inputSize x $inputSize")
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)
        Log.d("ImageClassifier", "ByteBuffer capacity: ${inputBuffer.capacity()}")

        // Create an output buffer with shape [1, 38]
        val outputBuffer = Array(1) { FloatArray(labels.size) }
        interpreter?.run(inputBuffer, outputBuffer)

        val probabilities = outputBuffer[0]
        Log.d("ImageClassifier", "Probabilities: " + probabilities.joinToString(separator = ", "))

        var maxIndex = 0
        var maxProb = probabilities[0]
        for (i in probabilities.indices) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i]
                maxIndex = i
            }
        }

        // Get the label corresponding to the highest probability
        val diseaseName = labels[maxIndex][0]
        val cause = labels[maxIndex][1]
        val treatment = labels[maxIndex][2]

        Log.d("ImageClassifier", "Predicted: $diseaseName (probability: $maxProb)")

        // Format the result as per the required output
        return "রোগ: $diseaseName\n\nকেন হয়: $cause\n\nপ্রতিকার: $treatment"
    }

    // Convert the bitmap to a ByteBuffer without additional normalization,
    // since your training data used raw pixel values (0–255).
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(inputSize * inputSize)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        // Optional: log raw pixel values of the first pixel.
        if (intValues.isNotEmpty()) {
            val firstPixel = intValues[0]
            val firstR = (firstPixel shr 16) and 0xFF
            val firstG = (firstPixel shr 8) and 0xFF
            val firstB = (firstPixel and 0xFF)
            Log.d("Preprocessing", "First raw pixel: R=$firstR, G=$firstG, B=$firstB")
        }

        // Write pixel values to the buffer as float values (range remains 0–255).
        for (pixel in intValues) {
            val r = ((pixel shr 16) and 0xFF).toFloat()
            val g = ((pixel shr 8) and 0xFF).toFloat()
            val b = (pixel and 0xFF).toFloat()
            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }
        return byteBuffer
    }
}
