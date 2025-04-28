package com.example.plant_disease_ver1

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var selectButton: Button
    private lateinit var captureButton: Button
    private lateinit var clearButton: Button
    private val PICK_IMAGE_REQUEST = 1
    private val CAPTURE_IMAGE_REQUEST = 2
    private lateinit var imageClassifier: ImageClassifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize views
        imageView = findViewById(R.id.imageView)
        resultText = findViewById(R.id.resultText)
        progressBar = findViewById(R.id.progressBar)
        selectButton = findViewById(R.id.selectButton)
        captureButton = findViewById(R.id.captureButton)
        clearButton = findViewById(R.id.clearButton)

        imageClassifier = ImageClassifier(this)

        // Select image button
        selectButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, PICK_IMAGE_REQUEST)
        }

        // Capture image button
        captureButton.setOnClickListener {
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(intent, CAPTURE_IMAGE_REQUEST)
        }

        // Clear button
        clearButton.setOnClickListener {
            imageView.setImageBitmap(null)
            resultText.text = "Prediction: "
            progressBar.visibility = ProgressBar.GONE
        }
    }

    // Handling image selection and prediction
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == RESULT_OK) {
            when (requestCode) {
                PICK_IMAGE_REQUEST -> {
                    val imageUri: Uri? = data?.data
                    val bitmap: Bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, imageUri)
                    imageView.setImageBitmap(bitmap)

                    progressBar.visibility = ProgressBar.VISIBLE
                    val prediction = imageClassifier.classify(bitmap)
                    progressBar.visibility = ProgressBar.GONE

                    resultText.text = prediction
                }

                CAPTURE_IMAGE_REQUEST -> {
                    val bitmap: Bitmap = data?.extras?.get("data") as Bitmap
                    imageView.setImageBitmap(bitmap)

                    progressBar.visibility = ProgressBar.VISIBLE
                    val prediction = imageClassifier.classify(bitmap)
                    progressBar.visibility = ProgressBar.GONE

                    resultText.text = prediction
                }
            }
        } else {
            Toast.makeText(this, "Failed to select image.", Toast.LENGTH_SHORT).show()
        }
    }
}
