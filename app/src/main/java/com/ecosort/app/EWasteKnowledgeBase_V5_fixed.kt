package com.ecosort.app

import android.content.Context
import org.json.JSONObject

class EWasteKnowledgeBase(context: Context) {

    private val appContext = context.applicationContext

    // Keys are normalized to lowercase with spaces converted to underscores.
    // Keep filenames aligned with the JSON files actually present in assets/e_waste.
    private val categoryMap = mapOf(
        "battery" to "batteries.json",
        "bulb" to "bulbs.json",
        "ic" to "integrated_circuits.json",
        "keyboard" to "keyboards.json",
        "laptop" to "laptops.json",
        "mouse" to "mice.json",
        "passive_components" to "passive_components.json",
        "pcb" to "pcbs.json",
        "phone" to "mobile_phones.json",
        "resistor" to "resistors.json",
        "wires" to "cables.json"
        // V5 also predicts "Transistor". No transistor knowledge filename
        // was established from the supplied files, so it safely falls back
        // to the unavailable-knowledge message until that JSON is added.
    )

    fun getKnowledge(predictedLabel: String): JSONObject? {
        val normalizedLabel = predictedLabel
            .trim()
            .lowercase()
            .replace(Regex("\\s+"), "_")

        val fileName = categoryMap[normalizedLabel] ?: return null

        return try {
            val jsonText = appContext.assets
                .open("e_waste/$fileName")
                .bufferedReader()
                .use { it.readText() }

            JSONObject(jsonText)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    fun buildLlamaContext(predictedLabel: String): String {
        val knowledge = getKnowledge(predictedLabel)

        if (knowledge == null) {
            return """
                The identified category is: $predictedLabel.

                Detailed safety and disposal information is not
                currently available in the knowledge base.

                Do not invent safety instructions or recycling
                information. Clearly tell the user that verified
                category-specific information is unavailable.
            """.trimIndent()
        }

        return """
            E-WASTE KNOWLEDGE BASE INFORMATION

            Item: ${knowledge.optString("item_name")}
            Hazard Level: ${knowledge.optString("hazard_level")}

            Hazards:
            ${knowledge.optJSONArray("hazards")}

            Safe Handling:
            ${knowledge.optJSONArray("safe_handling")}

            Unsafe Actions:
            ${knowledge.optJSONArray("unsafe_actions")}

            What To Do If Damaged:
            ${knowledge.optString("what_to_do_if_damaged")}

            Disposal Route:
            ${knowledge.optString("disposal_route")}

            Recoverable Materials:
            ${knowledge.optJSONArray("recoverable_materials")}

            Environmental Impact:
            ${knowledge.optString("environmental_impact")}
        """.trimIndent()
    }
}
