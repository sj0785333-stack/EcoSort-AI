
package com.ecosort.app

import android.os.Bundle
import android.widget.Toast
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.view.View
import android.view.Gravity
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import androidx.drawerlayout.widget.DrawerLayout
import androidx.core.view.GravityCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowCompat
import org.pytorch.executorch.extension.llm.LlmModule
import org.pytorch.executorch.extension.llm.LlmModuleConfig
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID
import java.util.concurrent.Executors
import android.net.Uri
import android.os.Environment
import android.widget.ImageView
import android.app.AlertDialog
import android.view.ViewGroup
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import android.graphics.BitmapFactory
import android.os.SystemClock
import android.os.Debug
import android.content.Intent
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import android.widget.PopupMenu

class MainActivity : AppCompatActivity() {

    private var llmModule: LlmModule? = null
    private lateinit var imageClassifier: EWasteImageClassifier
    private var classifierReady = false
    private var lastLlamaLoadTimeMs: Long? = null
    private val benchmarkPrefs by lazy {
        getSharedPreferences("EcoSortBenchmark", MODE_PRIVATE)
    }
    private lateinit var benchmarkExportLauncher: ActivityResultLauncher<String>

    // IMAGE CLASSIFICATION CONFIDENCE THRESHOLD
    private val classificationConfidenceThreshold = 60f

    private lateinit var knowledgeBase: EWasteKnowledgeBase
    // CAMERA VARIABLES

    private lateinit var cameraLauncher: ActivityResultLauncher<Uri>

    private lateinit var photoPreview: ImageView

    private var pendingPhotoUri: Uri? = null

    private var selectedPhotoUri: Uri? = null

    private val executor = Executors.newSingleThreadExecutor()

    private val historyPrefs by lazy {
        getSharedPreferences("EcoSortHistory", MODE_PRIVATE)
    }

    private var currentConversationId = UUID.randomUUID().toString()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        benchmarkExportLauncher = registerForActivityResult(
            ActivityResultContracts.CreateDocument("text/csv")
        ) { uri ->
            if (uri != null) {
                exportBenchmarkCsv(uri)
            }
        }

        // CAMERA RESULT HANDLER

        cameraLauncher = registerForActivityResult(
            ActivityResultContracts.TakePicture()
        ) { success ->

            val imageUri = pendingPhotoUri

            if (success && imageUri != null) {
                showPhotoPreview(imageUri)

            } else {
                Toast.makeText(
                    this,
                    "Photo capture cancelled",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
        WindowCompat.setDecorFitsSystemWindows(window, false)

        setContentView(R.layout.activity_main)

        // MAIN VIEWS

        // KEYBOARD AND SYSTEM BAR HANDLING

        val mainContent = findViewById<LinearLayout>(R.id.mainContent)

        val initialLeft = mainContent.paddingLeft
        val initialTop = mainContent.paddingTop
        val initialRight = mainContent.paddingRight
        val initialBottom = mainContent.paddingBottom

        ViewCompat.setOnApplyWindowInsetsListener(mainContent) { view, insets ->

            val systemBars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars()
            )

            val ime = insets.getInsets(
                WindowInsetsCompat.Type.ime()
            )

            val bottomInset = maxOf(systemBars.bottom, ime.bottom)

            view.setPadding(
                initialLeft + systemBars.left,
                initialTop + systemBars.top,
                initialRight + systemBars.right,
                initialBottom + bottomInset
            )

            insets
        }

        ViewCompat.requestApplyInsets(mainContent)

        // Keep the history drawer content clear of the status and navigation bars.
        val historyDrawer = findViewById<LinearLayout>(R.id.historyDrawer)
        val drawerLeft = historyDrawer.paddingLeft
        val drawerTop = historyDrawer.paddingTop
        val drawerRight = historyDrawer.paddingRight
        val drawerBottom = historyDrawer.paddingBottom

        ViewCompat.setOnApplyWindowInsetsListener(historyDrawer) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(
                drawerLeft + bars.left,
                drawerTop + bars.top,
                drawerRight + bars.right,
                drawerBottom + bars.bottom
            )
            insets
        }
        ViewCompat.requestApplyInsets(historyDrawer)

        val drawerLayout =
            findViewById<DrawerLayout>(R.id.rootLayout)

        val historyButton =
            findViewById<TextView>(R.id.historyButton)

        val historyList =
            findViewById<LinearLayout>(R.id.historyList)

        val searchHistory =
            findViewById<EditText>(R.id.searchHistory)

        val newChatButton =
            findViewById<Button>(R.id.newChatButton)

        val profileMenuButton =
            findViewById<TextView>(R.id.profileMenuButton)

        val userInput =
            findViewById<EditText>(R.id.userInput)

        val cameraButton =
            findViewById<TextView>(R.id.cameraButton)

        photoPreview = findViewById(R.id.photoPreview)

        photoPreview.setOnClickListener {

            selectedPhotoUri = null

            photoPreview.setImageDrawable(null)
            photoPreview.visibility = View.GONE

            Toast.makeText(
                this,
                "Photo removed",
                Toast.LENGTH_SHORT
            ).show()
        }

        cameraButton.setOnClickListener {
            openCamera()
        }

        val sendButton =
            findViewById<Button>(R.id.sendButton)

        val chatContainer =
            findViewById<LinearLayout>(R.id.chatContainer)

        val chatScrollView =
            findViewById<ScrollView>(R.id.chatScrollView)

        val welcomeSection =
            findViewById<LinearLayout>(R.id.welcomeSection)


        // LOAD HISTORY

        loadConversationHistory(historyList)


        // SEARCH HISTORY

        searchHistory.addTextChangedListener(
            object : android.text.TextWatcher {

                override fun beforeTextChanged(
                    s: CharSequence?,
                    start: Int,
                    count: Int,
                    after: Int
                ) {}

                override fun onTextChanged(
                    s: CharSequence?,
                    start: Int,
                    before: Int,
                    count: Int
                ) {
                    loadConversationHistory(
                        historyList,
                        s.toString()
                    )
                }

                override fun afterTextChanged(
                    s: android.text.Editable?
                ) {}
            }
        )


        // OPEN / CLOSE HISTORY DRAWER

        historyButton.setOnClickListener {

            if (drawerLayout.isDrawerOpen(GravityCompat.START)) {
                drawerLayout.closeDrawer(GravityCompat.START)
            } else {
                drawerLayout.openDrawer(GravityCompat.START)
            }
        }

        profileMenuButton.setOnClickListener {

            val popup = PopupMenu(this, profileMenuButton)

            popup.menu.add("Developer Access")
            popup.menu.add("About EcoSort")

            popup.setOnMenuItemClickListener { item ->

                when (item.title.toString()) {

                    "Developer Access" -> {
                        showDeveloperBenchmarkReport()
                        true
                    }

                    "About EcoSort" -> {

                        AlertDialog.Builder(this)
                            .setTitle("EcoSort AI")
                            .setMessage(
                                "Your Smart E-Waste Assistant\n\n" +
                                        "An offline AI-powered application " +
                                        "for electronic waste identification, " +
                                        "recycling and safe disposal."
                            )
                            .setPositiveButton("Close", null)
                            .show()

                        true
                    }

                    else -> false
                }
            }

            popup.show()
        }


        // SYSTEM INSETS

        // SYSTEM INSETS + KEYBOARD



        // KEYBOARD

        window.setSoftInputMode(
            WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE
        )


        // LOAD AI MODELS

        imageClassifier = EWasteImageClassifier(this)
        knowledgeBase = EWasteKnowledgeBase(this)

        loadLlamaModel()

//        executor.execute {
//            try {
//                imageClassifier.loadModel()
//                classifierReady = true
//
//                runOnUiThread {
//                    Toast.makeText(
//                        this,
//                        "EcoSort image classifier loaded!",
//                        Toast.LENGTH_LONG
//                    ).show()
//                }
//
//            } catch (e: Exception) {
//                e.printStackTrace()
//
//                runOnUiThread {
//                    Toast.makeText(
//                        this,
//                        "Classifier error: ${e.message}",
//                        Toast.LENGTH_LONG
//                    ).show()
//                }
//            }
//        }


        // SEND MESSAGE

        sendButton.setOnClickListener {

            val question = userInput.text.toString().trim()
            val imageUri = selectedPhotoUri

            // Accept text, photo, or both
            if (question.isEmpty() && imageUri == null) {
                Toast.makeText(
                    this,
                    "Please enter a question or attach a photo.",
                    Toast.LENGTH_SHORT
                ).show()
                return@setOnClickListener
            }



            val finalQuestion = question.ifEmpty {
                "Identify this e-waste item and explain its safe handling and disposal."
            }

            welcomeSection.visibility = View.GONE
            chatScrollView.visibility = View.VISIBLE

            userInput.text.clear()

            if (imageUri != null) {

                val photoStartTimeMs = SystemClock.elapsedRealtime()
                selectedPhotoUri = null
                photoPreview.setImageDrawable(null)
                photoPreview.visibility = View.GONE

                executor.execute {

                    try {
                        // STEP 1: Unload Llama to free memory
                        unloadLlamaModel()

                        // STEP 2: Load image classifier and measure model loading time
                        val classifierLoadStartTime = SystemClock.elapsedRealtime()
                        imageClassifier.loadModel()
                        val classifierLoadTimeMs =
                            SystemClock.elapsedRealtime() - classifierLoadStartTime

                        // STEP 3: Read captured image
                        val bitmap = contentResolver
                            .openInputStream(imageUri)
                            ?.use { BitmapFactory.decodeStream(it) }
                            ?: throw Exception("Could not read captured image")

                        // STEP 4: Measure image classification time
                        val inferenceStartTime = SystemClock.elapsedRealtime()

                        val result = imageClassifier.classify(bitmap)

                        val inferenceTimeMs =
                            SystemClock.elapsedRealtime() - inferenceStartTime
                        val category = result.category
                        val confidence = result.confidence * 100f
                        runOnUiThread {
                            Toast.makeText(
                                this,
                                "Detected: ${category.replace("_", " ")}\n" +
                                        "Confidence: ${"%.2f".format(confidence)}%",
                                Toast.LENGTH_LONG
                            ).show()
                        }

                        // STEP 5: Retrieve knowledge
                        val knowledge = knowledgeBase.getKnowledge(category)

                        val knowledgeContext = knowledge?.toString()
                            ?: "No verified category-specific knowledge is available for $category."

                        // STEP 6: Release classifier
                        imageClassifier.unloadModel()

                        // STEP 7: Reload Llama
                        loadLlamaModelBlocking()

                        // STEP 8: Prepare enhanced question
                        val enhancedQuestion = """
            The image classifier identified the photographed item as: $category.
            Classification confidence: ${"%.2f".format(confidence)}%.

            Relevant e-waste knowledge:
            $knowledgeContext

            User's question:
            $finalQuestion

            Explain the identification, relevant hazards, safe handling,
            and responsible disposal guidance using the supplied knowledge.
            Do not invent missing information.
        """.trimIndent()

                        // STEP 9: CHECK CLASSIFICATION CONFIDENCE
                        runOnUiThread {

                            if (confidence < classificationConfidenceThreshold) {

                                // Low confidence: Do not ask Llama to identify the item
                                showNotSureMessage(
                                    finalQuestion,
                                    confidence,
                                    chatContainer,
                                    imageUri
                                )
                                saveConversation(
                                    finalQuestion,
                                    "I couldn't confidently identify this electronic item. Classification confidence: ${"%.2f".format(confidence)}%. Please try another clear photograph with the entire component visible and good lighting. Avoid handling or disposing of an unidentified potentially hazardous component until its identity is confirmed.",
                                    imageUri
                                )
                                loadConversationHistory(
                                    findViewById(R.id.historyList),
                                    findViewById<EditText>(R.id.searchHistory).text.toString()
                                )
                                saveBenchmarkSnapshot(
                                    classifierLoadTimeMs = classifierLoadTimeMs,
                                    classifierInferenceTimeMs = inferenceTimeMs,
                                    llamaLoadTimeMs = lastLlamaLoadTimeMs,
                                    generationTimeMs = null,
                                    totalResponseTimeMs =
                                        SystemClock.elapsedRealtime() - photoStartTimeMs,
                                    memoryUsageMb = Debug.getPss() / 1024f,
                                    note = "Llama generation skipped: low classification confidence"
                                )

                            } else {

                                // Sufficient confidence: Continue normal AI response
                                generateResponse(
                                    finalQuestion,
                                    chatContainer,
                                    enhancedQuestion,
                                    category,
                                    confidence,
                                    inferenceTimeMs,
                                    classifierLoadTimeMs,
                                    photoStartTimeMs,
                                    imageUri
                                )
                            }
                        }



                    } catch (e: Exception) {

                        e.printStackTrace()

                        // Release classifier if it was loaded
                        try {
                            imageClassifier.unloadModel()
                        } catch (_: Exception) {
                        }

                        // Restore Llama if necessary
                        if (llmModule == null) {
                            try {
                                loadLlamaModelBlocking()
                            } catch (loadError: Exception) {
                                loadError.printStackTrace()
                            }
                        }

                        runOnUiThread {
                            Toast.makeText(
                                this,
                                "Image processing failed: ${e.message}",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    }
                }


            } else {

                generateResponse(
                    finalQuestion,
                    chatContainer
                )
            }
        }


        // NEW CHAT

        newChatButton.setOnClickListener {

            chatContainer.removeAllViews()

            chatScrollView.visibility = View.GONE
            welcomeSection.visibility = View.VISIBLE

            currentConversationId = UUID.randomUUID().toString()

            userInput.text.clear()
            searchHistory.text.clear()

            drawerLayout.closeDrawer(GravityCompat.START)
        }
    }
    // Release Llama native resources
    private fun unloadLlamaModel() {
        llmModule?.close()
        llmModule = null
    }


    // --------------------------------------------------
    // LOAD LLAMA MODEL
    // --------------------------------------------------

    private fun loadLlamaModel() {
        executor.execute {
            loadLlamaModelBlocking()
        }
    }

    private fun loadLlamaModelBlocking() {
        try {
            val modelDirectory = File(
                getExternalFilesDir(null),
                "llama"
            )

            val modelFile = File(
                modelDirectory,
                "Llama-3.2-1B-Instruct-8da4w-gptq-2k.pte"
            )

            val tokenizerFile = File(
                modelDirectory,
                "tokenizer.json"
            )

            if (!modelFile.exists() || !tokenizerFile.exists()) {
                throw Exception("Model or tokenizer file missing")
            }

            val config = LlmModuleConfig.create()
                .modulePath(modelFile.absolutePath)
                .tokenizerPath(tokenizerFile.absolutePath)
                .temperature(0.8f)
                .modelType(LlmModuleConfig.MODEL_TYPE_TEXT)
                .loadMode(LlmModuleConfig.LOAD_MODE_MMAP)
                .build()

            val module = LlmModule(config)
            val llamaLoadStartTime = SystemClock.elapsedRealtime()
            module.load()
            lastLlamaLoadTimeMs =
                SystemClock.elapsedRealtime() - llamaLoadStartTime

            llmModule = module

            runOnUiThread {
                Toast.makeText(
                    this,
                    "Llama model loaded successfully!",
                    Toast.LENGTH_SHORT
                ).show()
            }

        } catch (e: Exception) {
            e.printStackTrace()

            runOnUiThread {
                Toast.makeText(
                    this,
                    "Llama Error: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            }

            throw e
        }
    }


    // --------------------------------------------------
    // GENERATE AI RESPONSE
    // --------------------------------------------------

    private fun generateResponse(
        question: String,
        chatContainer: LinearLayout,
        additionalContext: String = "",
        classificationCategory: String? = null,
        classificationConfidence: Float? = null,
        classificationInferenceTimeMs: Long? = null,
        classifierLoadTimeMs: Long? = null,
        photoStartTimeMs: Long? = null,
        photoUri: Uri? = null
    ){

        val module = llmModule

        if (module == null) {

            Toast.makeText(
                this,
                "Model is still loading. Please wait.",
                Toast.LENGTH_SHORT
            ).show()

            return
        }

        val chatScrollView =
            findViewById<ScrollView>(R.id.chatScrollView)

        val prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are EcoSort AI, a practical, friendly and conversational e-waste
identification, safety and recycling assistant.

YOUR PURPOSE:
Help the user understand the item, make a safe decision, and know what
practical step to take next. Be an interactive assistant, not a generic
report generator.

HOW TO ANSWER:
- Start with a direct answer to the user's question.
- Use the detected item and supplied knowledge when available.
- Prioritise the most relevant hazards and practical actions for THIS item.
- Give clear, actionable steps for safe handling, reuse or disposal.
- Mention environmental impact only when it adds useful information.
- Adapt the amount of detail to the question. A simple question deserves
  a short answer; a safety or disposal question may need more detail.
- Avoid repeating the full identification in every follow-up.
- Resolve references such as "it" using the conversation context.
- If the item cannot be identified reliably, say so rather than guessing.

ACCURACY AND SAFETY:
- Treat the supplied knowledge base as the primary category-specific source.
- Never invent facts, statistics, recycling centres, collection services,
  laws or regulations.
- Clearly say when the supplied information does not establish an answer.
- Do not infer that an item is safe just because it looks undamaged.
- For batteries, distinguish general precautions from chemistry-specific advice.
- Never recommend burning, puncturing, crushing, opening or dismantling batteries.
- For damaged, swollen, leaking or unusually hot batteries, advise the user
  to avoid unnecessary handling and seek help from an appropriate authorised
  collection or safety service.
- Do not give hazardous DIY repair or material-recovery instructions.
- Use simple, professional, natural English. Avoid robotic repetition,
  excessive headings and vague environmental statements.

IMPORTANT:
- First complete the actual answer to the user's question.
- Do not generate follow-up questions in this response.
- Do not output markers, metadata, or instructions for the application.

SUPPLIED KNOWLEDGE AND CONVERSATION CONTEXT:
$additionalContext

Use supplied information as your primary reference. Do not invent missing
category-specific information.

<|eot_id|><|start_header_id|>user<|end_header_id|>

$question

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""".trimIndent()
        val maxBubbleWidth =
            (resources.displayMetrics.widthPixels * 0.78f).toInt()


        // USER PHOTO (when this turn includes a captured image)
        if (photoUri != null) {
            addUserPhotoBubble(photoUri, chatContainer)
        }

        // USER MESSAGE
        val userBubble = createChatBubble(question, isUser = true, maxWidth = maxBubbleWidth)
        val userParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            gravity = Gravity.END
            setMargins(8.dp(), 0, 8.dp(), 16.dp())
        }
        chatContainer.addView(userBubble, userParams)

        // SHOW CLASSIFICATION RESULT AFTER USER MESSAGE
        if (classificationCategory != null &&
            classificationConfidence != null
        ) {
            addClassificationCard(
                classificationCategory,
                classificationConfidence,
                chatContainer
            )
        }


        // AI MESSAGE
        val aiBubble = createChatBubble("Thinking...", isUser = false, maxWidth = maxBubbleWidth)
        val aiParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            gravity = Gravity.START
            setMargins(8.dp(), 0, 8.dp(), 20.dp())
        }
        chatContainer.addView(aiBubble, aiParams)


        chatScrollView.post {
            chatScrollView.fullScroll(View.FOCUS_DOWN)
        }


        // GENERATION

        executor.execute {

            val fullAnswer = StringBuilder()

            val generationConfig =
                LlmGenerationConfig.create()
                    .seqLen(2048)
                    .maxNewTokens(256)
                    .echo(false)
                    .build()

            // Each user turn is a fresh prompt that includes only the
            // context we deliberately pass in. Clear stale KV-cache tokens
            // left by the previous generation before starting this turn.
            module.resetContext()

            val generationStartTime = SystemClock.elapsedRealtime()

            module.generate(
                prompt,
                generationConfig,
                object : LlmCallback {

                    override fun onResult(token: String) {
                        // Accumulate tokens in the background. The answer is
                        // displayed only once, after generation has completed.
                        fullAnswer.append(token.replace("<|eot_id|>", ""))
                    }


                    override fun onStats(statsJson: String) {

                        // Measure until the generation-complete callback.
                        val generationTimeMs =
                            SystemClock.elapsedRealtime() - generationStartTime
                        val totalPhotoResponseTimeMs = photoStartTimeMs?.let {
                            SystemClock.elapsedRealtime() - it
                        }
                        val memoryUsageMb = Debug.getPss() / 1024f

                        val answer = formatAiResponse(fullAnswer.toString())

                        if (answer.isNotEmpty()) {
                            // Show and save the complete answer FIRST.
                            runOnUiThread {
                                aiBubble.text = answer
                                chatScrollView.post {
                                    chatScrollView.fullScroll(View.FOCUS_DOWN)
                                }
                            }

                            saveConversation(question, answer, photoUri)

                            runOnUiThread {
                                loadConversationHistory(
                                    findViewById(R.id.historyList),
                                    findViewById<EditText>(R.id.searchHistory).text.toString()
                                )
                            }

                            // Only after the answer is complete, ask Llama for
                            // three contextual follow-up questions in a second call.
                            executor.execute {
                                generateFollowUpSuggestions(
                                    module = module,
                                    previousQuestion = question,
                                    previousAnswer = answer,
                                    previousContext = additionalContext,
                                    chatContainer = chatContainer
                                )
                            }
                        }

                        saveBenchmarkSnapshot(
                            classifierLoadTimeMs = classifierLoadTimeMs,
                            classifierInferenceTimeMs = classificationInferenceTimeMs,
                            llamaLoadTimeMs = lastLlamaLoadTimeMs,
                            generationTimeMs = generationTimeMs,
                            totalResponseTimeMs = totalPhotoResponseTimeMs,
                            memoryUsageMb = memoryUsageMb,
                            note = if (photoStartTimeMs != null) "Photo workflow" else "Text-only query"
                        )

                        runOnUiThread {

                            chatScrollView.post {
                                chatScrollView.fullScroll(
                                    View.FOCUS_DOWN
                                )
                            }
                        }
                    }


                    override fun onError(
                        errorCode: Int,
                        message: String
                    ) {

                        runOnUiThread {

                            aiBubble.text =
                                "Sorry, something went wrong.\n$message"
                        }
                    }
                }
            )
        }
    }

    // Generate suggestions in a separate call so the main answer is always
    // completed and displayed before follow-up questions are requested.
    private fun generateFollowUpSuggestions(
        module: LlmModule,
        previousQuestion: String,
        previousAnswer: String,
        previousContext: String,
        chatContainer: LinearLayout
    ) {
        // Keep the suggestion prompt compact for the model's 2K-token context.
        val compactContext = previousContext.take(1600)
        val compactAnswer = previousAnswer.take(1800)

        val followUpPrompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You generate useful follow-up questions for EcoSort AI, an e-waste safety assistant.
Generate exactly THREE short, specific questions that naturally follow the answer.
Questions must relate to the identified item and help the user decide what to do next.
Do not repeat the original question. Do not answer the questions.
Output only the three questions, one per line. No heading, marker, numbering or bullets.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Item and knowledge context:
$compactContext

Previous user question:
$previousQuestion

EcoSort AI answer:
$compactAnswer
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""".trimIndent()

        val generatedQuestions = StringBuilder()
        val followUpConfig = LlmGenerationConfig.create()
            .seqLen(2048)
            .maxNewTokens(96)
            .echo(false)
            .build()

        try {
            // The previous answer's KV cache must not be appended to this
            // separate suggestion request.
            module.resetContext()

            module.generate(
                followUpPrompt,
                followUpConfig,
                object : LlmCallback {
                    override fun onResult(token: String) {
                        generatedQuestions.append(token.replace("<|eot_id|>", ""))
                    }

                    override fun onStats(statsJson: String) {
                        val questions = generatedQuestions.toString()
                            .lineSequence()
                            .map { line ->
                                line.trim()
                                    .replace(Regex("^(?:[-•*]\\s*|\\d+[.)]\\s*)"), "")
                                    .trim()
                            }
                            .filter { it.endsWith("?") && it.length in 8..140 }
                            .distinct()
                            .take(3)
                            .toList()

                        runOnUiThread {
                            addFollowUpQuestions(
                                chatContainer = chatContainer,
                                questions = questions,
                                previousQuestion = previousQuestion,
                                previousAnswer = previousAnswer,
                                previousContext = previousContext
                            )
                        }
                    }

                    override fun onError(errorCode: Int, message: String) {
                        runOnUiThread {
                            addFollowUpQuestions(
                                chatContainer = chatContainer,
                                questions = emptyList(),
                                previousQuestion = previousQuestion,
                                previousAnswer = previousAnswer,
                                previousContext = previousContext
                            )
                        }
                    }
                }
            )
        } catch (_: Exception) {
            runOnUiThread {
                addFollowUpQuestions(
                    chatContainer = chatContainer,
                    questions = emptyList(),
                    previousQuestion = previousQuestion,
                    previousAnswer = previousAnswer,
                    previousContext = previousContext
                )
            }
        }
    }

    // Display contextual, tappable follow-up questions below the assistant's answer.
    private fun addFollowUpQuestions(
        chatContainer: LinearLayout,
        questions: List<String>,
        previousQuestion: String,
        previousAnswer: String,
        previousContext: String
    ) {
        val usableQuestions = if (questions.isNotEmpty()) questions else listOf(
            "What should I do next with this item?",
            "How can I handle or store it safely?",
            "What is the proper way to recycle it?"
        )

        val followUpContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(8.dp(), 0, 8.dp(), 8.dp())
        }

        val label = TextView(this).apply {
            text = "YOU CAN ALSO ASK"
            textSize = 12f
            setTextColor(Color.rgb(0, 91, 70))
            typeface = Typeface.DEFAULT_BOLD
            setPadding(8.dp(), 4.dp(), 8.dp(), 8.dp())
        }
        followUpContainer.addView(label)

        usableQuestions.forEach { followUpQuestion ->
            val followUpButton = Button(this).apply {
                text = followUpQuestion
                isAllCaps = false
                textSize = 14f
                setTextColor(Color.rgb(0, 91, 70))
                typeface = Typeface.SERIF
                setPadding(14.dp(), 8.dp(), 14.dp(), 8.dp())
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = 22.dp().toFloat()
                    setColor(Color.rgb(232, 241, 232))
                    setStroke(1.dp(), Color.rgb(0, 91, 70))
                }
                stateListAnimator = null
            }

            val buttonParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = 7.dp() }

            followUpButton.setOnClickListener {
                // Carry forward only a compact, relevant summary of the last
                // exchange. The full prior KV cache is cleared before generation.
                val contextForFollowUp = buildString {
                    append(previousContext.take(1600))
                    append("\n\nPrevious user question: ")
                    append(previousQuestion.take(300))
                    append("\nPrevious EcoSort AI answer: ")
                    append(previousAnswer.take(1800))
                    append("\nAnswer the selected follow-up directly. Do not restart identification.")
                }
                generateResponse(
                    question = followUpQuestion,
                    chatContainer = chatContainer,
                    additionalContext = contextForFollowUp
                )
            }
            followUpContainer.addView(followUpButton, buttonParams)
        }

        val containerParams = LinearLayout.LayoutParams(
            (resources.displayMetrics.widthPixels * 0.86f).toInt(),
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            gravity = Gravity.START
            setMargins(8.dp(), 0, 8.dp(), 18.dp())
        }
        chatContainer.addView(followUpContainer, containerParams)
        findViewById<ScrollView>(R.id.chatScrollView).post {
            findViewById<ScrollView>(R.id.chatScrollView).fullScroll(View.FOCUS_DOWN)
        }
    }

    // FORMAT AI RESPONSE
    private fun formatAiResponse(response: String): String {

        return response
            .replace("**", "")
            .replace(Regex("(?m)^#{1,6}\\s*"), "")
            .replace(Regex("(?m)^[-*]\\s+"), "• ")
            .replace(Regex("\\n{3,}"), "\n\n")
            .trim()
    }

    // Add a captured photo to the chat as a right-aligned user message.
    private fun addUserPhotoBubble(uri: Uri, chatContainer: LinearLayout) {
        val photo = ImageView(this).apply {
            scaleType = ImageView.ScaleType.CENTER_CROP
            contentDescription = "Photo sent to EcoSort AI"
            background = GradientDrawable().apply {
                cornerRadius = 18.dp().toFloat()
                setColor(Color.rgb(232, 241, 232))
            }
            clipToOutline = true

            try {
                setImageURI(uri)
            } catch (_: Exception) {
                setImageDrawable(null)
            }
        }

        val params = LinearLayout.LayoutParams(
            220.dp(),
            180.dp()
        ).apply {
            gravity = Gravity.END
            setMargins(8.dp(), 0, 8.dp(), 8.dp())
        }
        chatContainer.addView(photo, params)
    }

    // Shared styling keeps user and assistant bubbles consistent in every chat view.
    private fun createChatBubble(
        message: String,
        isUser: Boolean,
        maxWidth: Int = (resources.displayMetrics.widthPixels * 0.78f).toInt()
    ): TextView {
        return TextView(this).apply {
            text = message
            textSize = 16f
            setTextColor(if (isUser) Color.WHITE else Color.rgb(25, 65, 48))
            typeface = Typeface.SERIF
            setPadding(18.dp(), 14.dp(), 18.dp(), 14.dp())
            this.maxWidth = maxWidth
            setTextIsSelectable(!isUser)

            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                val large = 24.dp().toFloat()
                val small = 7.dp().toFloat()
                cornerRadii = if (isUser) {
                    floatArrayOf(large, large, large, large, small, small, large, large)
                } else {
                    floatArrayOf(small, small, large, large, large, large, large, large)
                }
                setColor(
                    if (isUser) Color.rgb(0, 91, 70)
                    else Color.rgb(245, 240, 230)
                )
            }
        }
    }


    // DISPLAY LOW-CONFIDENCE RESULT
    private fun showNotSureMessage(
        question: String,
        confidence: Float,
        chatContainer: LinearLayout,
        photoUri: Uri? = null
    ) {

        val maxBubbleWidth =
            (resources.displayMetrics.widthPixels * 0.78f).toInt()

        // User photo
        if (photoUri != null) {
            addUserPhotoBubble(photoUri, chatContainer)
        }

        // User message
        val userBubble = createChatBubble(question, isUser = true, maxWidth = maxBubbleWidth)
        val userParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            gravity = Gravity.END
            setMargins(8.dp(), 0, 8.dp(), 16.dp())
        }
        chatContainer.addView(userBubble, userParams)

        // Assistant message for low-confidence classification
        val notSureText = """
            NOT SURE

            I couldn't confidently identify this electronic item.

            Classification confidence: ${"%.2f".format(confidence)}%

            Please try taking another clear photograph with the
            entire component visible and good lighting.

            For safety, avoid handling or disposing of an
            unidentified or potentially hazardous component
            until its identity is confirmed.
        """.trimIndent()

        val notSureBubble = createChatBubble(
            notSureText,
            isUser = false,
            maxWidth = maxBubbleWidth
        )
        val aiParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            gravity = Gravity.START
            setMargins(8.dp(), 0, 8.dp(), 20.dp())
        }
        chatContainer.addView(notSureBubble, aiParams)

        findViewById<ScrollView>(R.id.chatScrollView).post {
            findViewById<ScrollView>(R.id.chatScrollView)
                .fullScroll(View.FOCUS_DOWN)
        }
    }


    // CLASSIFICATION RESULT CARD
    private fun addClassificationCard(
        category: String,
        confidence: Float,
        chatContainer: LinearLayout
    ) {

        val card = TextView(this).apply {

            text = """
    ECOSORT IDENTIFICATION

    Detected item: ${category.replace("_", " ").uppercase()}

    Confidence: ${"%.2f".format(confidence)}%

""".trimIndent()


            textSize = 15f
            setTextColor(Color.rgb(25, 65, 48))
            typeface = Typeface.SERIF

            setPadding(24, 22, 24, 22)

            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = 24f
                setColor(Color.rgb(232, 241, 232))
                setStroke(2, Color.rgb(0, 91, 70))
            }
        }

        val params = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            setMargins(16, 12, 16, 16)
        }

        chatContainer.addView(card, params)
    }


    // --------------------------------------------------
    // PRIVATE PERFORMANCE BENCHMARK STORAGE
    // --------------------------------------------------

    private fun saveBenchmarkSnapshot(
        classifierLoadTimeMs: Long?,
        classifierInferenceTimeMs: Long?,
        llamaLoadTimeMs: Long?,
        generationTimeMs: Long?,
        totalResponseTimeMs: Long?,
        memoryUsageMb: Float?,
        note: String
    ) {
        val metrics = JSONObject().apply {
            put("classifierLoadTimeMs", classifierLoadTimeMs ?: JSONObject.NULL)
            put("classifierInferenceTimeMs", classifierInferenceTimeMs ?: JSONObject.NULL)
            put("llamaLoadTimeMs", llamaLoadTimeMs ?: JSONObject.NULL)
            put("generationTimeMs", generationTimeMs ?: JSONObject.NULL)
            put("totalResponseTimeMs", totalResponseTimeMs ?: JSONObject.NULL)
            put("memoryUsageMb", memoryUsageMb ?: JSONObject.NULL)
            put("note", note)
            put("timestamp", System.currentTimeMillis())
        }

        val history = try {
            JSONArray(benchmarkPrefs.getString("benchmarkHistory", "[]"))
        } catch (_: Exception) {
            JSONArray()
        }
        history.put(metrics)

        benchmarkPrefs.edit()
            .putString("lastBenchmark", metrics.toString())
            .putString("benchmarkHistory", history.toString())
            .apply()
    }

    // Developer-only report: open by long-pressing the history/menu button.
    private fun showDeveloperBenchmarkReport() {
        val rawMetrics = benchmarkPrefs.getString("lastBenchmark", null)

        if (rawMetrics.isNullOrBlank()) {
            AlertDialog.Builder(this)
                .setTitle("Developer Benchmark")
                .setMessage("No benchmark data yet. Run an image classification or text query first.")
                .setPositiveButton("OK", null)
                .show()
            return
        }

        val metrics = try {
            JSONObject(rawMetrics)
        } catch (_: Exception) {
            AlertDialog.Builder(this)
                .setTitle("Developer Benchmark")
                .setMessage("Saved benchmark data could not be read.")
                .setPositiveButton("OK", null)
                .show()
            return
        }

        fun displayValue(key: String, suffix: String = " ms"): String {
            val value = metrics.opt(key)
            if (value == null || value == JSONObject.NULL) return "Not available"
            return if (key == "memoryUsageMb") {
                "${String.format(Locale.US, "%.1f", (value as Number).toDouble())}$suffix"
            } else {
                "$value$suffix"
            }
        }

        val timestamp = metrics.optLong("timestamp", 0L)
        val readableTimestamp = if (timestamp > 0L) {
            SimpleDateFormat("dd MMM yyyy, hh:mm:ss a", Locale.getDefault())
                .format(Date(timestamp))
        } else {
            "Not available"
        }

        val report = """
            Last run: ${metrics.optString("note", "Benchmark")}

            Classifier load: ${displayValue("classifierLoadTimeMs")}
            Classification inference: ${displayValue("classifierInferenceTimeMs")}
            Llama load: ${displayValue("llamaLoadTimeMs")}
            Llama generation: ${displayValue("generationTimeMs")}
            Photo-to-answer: ${displayValue("totalResponseTimeMs")}
            App memory (PSS): ${displayValue("memoryUsageMb", " MB")}

            Timestamp: $readableTimestamp
        """.trimIndent()

        AlertDialog.Builder(this)
            .setTitle("Developer Benchmark")
            .setMessage(report)
            .setPositiveButton("Export CSV") { _, _ ->
                val fileName = "EcoSort_Benchmark_${System.currentTimeMillis()}.csv"
                benchmarkExportLauncher.launch(fileName)
            }
            .setNeutralButton("Clear data") { _, _ ->
                benchmarkPrefs.edit()
                    .remove("lastBenchmark")
                    .remove("benchmarkHistory")
                    .apply()
                Toast.makeText(this, "Benchmark history cleared", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Close", null)
            .show()
    }

    private fun exportBenchmarkCsv(uri: Uri) {
        val history = try {
            JSONArray(benchmarkPrefs.getString("benchmarkHistory", "[]"))
        } catch (_: Exception) {
            JSONArray()
        }

        if (history.length() == 0) {
            Toast.makeText(this, "No benchmark history to export", Toast.LENGTH_SHORT).show()
            return
        }

        val header = listOf(
            "Run", "Timestamp", "Run Type", "Classifier Load (ms)",
            "Classification Inference (ms)", "Llama Load (ms)",
            "Llama Generation (ms)", "Photo-to-answer (ms)", "App Memory PSS (MB)"
        )
        fun csvValue(value: Any?): String {
            val text = value?.toString().orEmpty()
            return "\"${text.replace("\"", "\"\"")}\""
        }

        try {
            contentResolver.openOutputStream(uri)?.bufferedWriter()?.use { writer ->
                writer.appendLine(header.joinToString(",") { csvValue(it) })
                for (index in 0 until history.length()) {
                    val item = history.optJSONObject(index) ?: continue
                    val timestamp = item.optLong("timestamp", 0L)
                    val readable = if (timestamp > 0L) {
                        SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
                            .format(Date(timestamp))
                    } else ""
                    val row = listOf(
                        (index + 1).toString(),
                        readable,
                        item.optString("note", ""),
                        item.opt("classifierLoadTimeMs"),
                        item.opt("classifierInferenceTimeMs"),
                        item.opt("llamaLoadTimeMs"),
                        item.opt("generationTimeMs"),
                        item.opt("totalResponseTimeMs"),
                        item.opt("memoryUsageMb")
                    )
                    writer.appendLine(row.joinToString(",") { csvValue(it) })
                }
            } ?: throw IllegalStateException("Could not open the selected file")

            Toast.makeText(this, "Benchmark CSV exported", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Toast.makeText(this, "CSV export failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }


    // --------------------------------------------------
    // SAVE CONVERSATION
    // --------------------------------------------------

    private fun saveConversation(
        question: String,
        answer: String,
        photoUri: Uri? = null
    ) {

        val conversations = JSONArray(
            historyPrefs.getString("conversations", "[]")
        )

        val conversation = JSONObject().apply {

            put("id", currentConversationId)
            put("title", question)
            put("question", question)
            put("answer", answer)
            put("photoUri", photoUri?.toString() ?: "")
            put("timestamp", System.currentTimeMillis())
        }

        conversations.put(conversation)

        historyPrefs.edit()
            .putString("conversations", conversations.toString())
            .apply()
    }


    // --------------------------------------------------
    // LOAD CONVERSATION HISTORY
    // --------------------------------------------------

    private fun loadConversationHistory(
        historyList: LinearLayout,
        searchText: String = ""
    ) {
        historyList.removeAllViews()

        val conversations = JSONArray(
            historyPrefs.getString("conversations", "[]")
        )

        for (i in conversations.length() - 1 downTo 0) {
            val item = conversations.getJSONObject(i)
            val title = item.optString("title", "New conversation")
            val timestamp = item.optLong("timestamp", 0L)
            val conversationId = item.optString("id", "")
            val question = item.optString("question", "")

            if (!title.contains(searchText, ignoreCase = true)) continue

            // Each existing saved question/answer entry gets its own row.
            val historyRow = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(4, 2, 4, 2)
            }

            val historyItem = TextView(this).apply {
                text = title
                textSize = 15f
                setTextColor(Color.WHITE)
                typeface = Typeface.SERIF
                setPadding(12, 16, 8, 16)
                isClickable = true
                isFocusable = true
                maxLines = 2
                ellipsize = android.text.TextUtils.TruncateAt.END
                contentDescription = "Open conversation: $title"
            }

            historyRow.addView(
                historyItem,
                LinearLayout.LayoutParams(
                    0,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    1f
                )
            )

            val deleteButton = TextView(this).apply {
                text = "Delete"
                textSize = 12f
                gravity = Gravity.CENTER
                setTextColor(Color.WHITE)
                setPadding(12, 10, 12, 10)
                background = GradientDrawable().apply {
                    cornerRadius = 12.dp().toFloat()
                    setColor(Color.rgb(120, 55, 48))
                }
                isClickable = true
                isFocusable = true
                contentDescription = "Delete this saved chat entry"

                setOnClickListener {
                    AlertDialog.Builder(this@MainActivity)
                        .setTitle("Delete chat?")
                        .setMessage("Delete this conversation and all its saved messages from your history?")
                        .setNegativeButton("Cancel", null)
                        .setPositiveButton("Delete") { _, _ ->
                            val latest = JSONArray(
                                historyPrefs.getString("conversations", "[]")
                            )
                            val updated = JSONArray()
                            var removed = false

                            for (index in 0 until latest.length()) {
                                val saved = latest.optJSONObject(index) ?: continue
                                val sameConversation = if (conversationId.isNotBlank()) {
                                    saved.optString("id", "") == conversationId
                                } else {
                                    saved.optLong("timestamp", -1L) == timestamp &&
                                            saved.optString("question", "") == question
                                }

                                if (sameConversation) {
                                    removed = true
                                } else {
                                    updated.put(saved)
                                }
                            }

                            historyPrefs.edit()
                                .putString("conversations", updated.toString())
                                .apply()

                            loadConversationHistory(
                                historyList,
                                searchText
                            )

                            Toast.makeText(
                                this@MainActivity,
                                if (removed) "Conversation deleted" else "Conversation was already removed",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                        .show()
                }
            }

            historyRow.addView(
                deleteButton,
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                )
            )

            historyItem.setOnClickListener {
                val answer = item.optString("answer", "")
                val chatContainer = findViewById<LinearLayout>(R.id.chatContainer)
                val chatScrollView = findViewById<ScrollView>(R.id.chatScrollView)
                val welcomeSection = findViewById<LinearLayout>(R.id.welcomeSection)

                chatContainer.removeAllViews()
                welcomeSection.visibility = View.GONE
                chatScrollView.visibility = View.VISIBLE

                val savedPhotoUri = item.optString("photoUri", "")
                    .takeIf { it.isNotBlank() }
                    ?.let { Uri.parse(it) }
                if (savedPhotoUri != null) {
                    addUserPhotoBubble(savedPhotoUri, chatContainer)
                }

                val maxBubbleWidth =
                    (resources.displayMetrics.widthPixels * 0.78f).toInt()

                val userMessage = createChatBubble(
                    question,
                    isUser = true,
                    maxWidth = maxBubbleWidth
                )
                val userParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    gravity = Gravity.END
                    setMargins(8.dp(), 0, 8.dp(), 16.dp())
                }
                chatContainer.addView(userMessage, userParams)

                val aiMessage = createChatBubble(
                    answer,
                    isUser = false,
                    maxWidth = maxBubbleWidth
                )
                val aiParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    gravity = Gravity.START
                    setMargins(8.dp(), 0, 8.dp(), 20.dp())
                }
                chatContainer.addView(aiMessage, aiParams)

                chatScrollView.post {
                    chatScrollView.fullScroll(View.FOCUS_DOWN)
                }

                findViewById<DrawerLayout>(R.id.rootLayout)
                    .closeDrawer(GravityCompat.START)
            }

            historyList.addView(historyRow)
        }
    }

    // --------------------------------------------------
    // CLEANUP
    // --------------------------------------------------
// --------------------------------------------------
// OPEN CAMERA
// --------------------------------------------------

    private fun openCamera() {

        try {
            val imageDirectory =
                getExternalFilesDir(Environment.DIRECTORY_PICTURES)
                    ?: throw IllegalStateException("Picture storage unavailable")

            if (!imageDirectory.exists()) {
                imageDirectory.mkdirs()
            }

            val photoFile = File.createTempFile(
                "EcoSort_",
                ".jpg",
                imageDirectory
            )

            val photoUri = FileProvider.getUriForFile(
                this,
                "$packageName.fileprovider",
                photoFile
            )

            pendingPhotoUri = photoUri

            cameraLauncher.launch(photoUri)

        } catch (e: Exception) {

            Toast.makeText(
                this,
                "Camera error: ${e.message}",
                Toast.LENGTH_LONG
            ).show()

            e.printStackTrace()
        }
    }


// --------------------------------------------------
// PHOTO PREVIEW
// --------------------------------------------------

    private fun showPhotoPreview(uri: Uri) {

        val imageView = ImageView(this).apply {

            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                280.dp()
            )

            scaleType = ImageView.ScaleType.FIT_CENTER

            setImageURI(uri)
        }

        val previewLayout = LinearLayout(this).apply {

            orientation = LinearLayout.VERTICAL

            setPadding(
                20.dp(),
                12.dp(),
                20.dp(),
                8.dp()
            )

            addView(imageView)
        }

        AlertDialog.Builder(this)

            .setTitle("Preview E-Waste Image")

            .setView(previewLayout)

            .setPositiveButton("Use Photo") { _, _ ->

                selectedPhotoUri = uri

                photoPreview.setImageURI(uri)
                photoPreview.visibility = View.VISIBLE

                Toast.makeText(
                    this,
                    "Photo attached successfully!",
                    Toast.LENGTH_SHORT
                ).show()
            }

            .setNegativeButton("Retake") { _, _ ->

                openCamera()
            }

            .setNeutralButton("Cancel", null)

            .show()
    }


// --------------------------------------------------
// DP CONVERSION
// --------------------------------------------------

    private fun Int.dp(): Int {

        return (this * resources.displayMetrics.density).toInt()
    }

    override fun onDestroy() {

        executor.shutdown()

        super.onDestroy()
    }
}
