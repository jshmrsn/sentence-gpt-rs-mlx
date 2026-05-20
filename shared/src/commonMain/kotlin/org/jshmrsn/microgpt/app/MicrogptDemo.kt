package org.jshmrsn.microgpt.app

import microgpt_kotlin_visualized.shared.generated.resources.Res
import org.jetbrains.compose.resources.ExperimentalResourceApi
import org.jshmrsn.microgpt.lib.TrainedMicrogpt
import org.jshmrsn.microgpt.lib.generateSamples
import org.jshmrsn.microgpt.lib.microgpt
import kotlin.random.Random

@OptIn(ExperimentalResourceApi::class)
suspend fun loadMicrogptInputText(): String =
    Res.readBytes("files/input.txt").decodeToString()

suspend fun trainMicrogptDemo(randomNumberGenerator: Random = Random(0)): TrainedMicrogpt {
    val inputText = loadMicrogptInputText()
    return microgpt(inputText, randomNumberGenerator)
}

fun generateMicrogptSamples(
    trainedMicrogpt: TrainedMicrogpt,
    prefix: String,
    sampleCount: Int,
    temperature: Double,
    randomNumberGenerator: Random
): List<String> =
    generateSamples(
        trainedMicrogpt = trainedMicrogpt,
        prefix = prefix,
        sampleCount = sampleCount,
        temperature = temperature,
        randomNumberGenerator = randomNumberGenerator
    )
