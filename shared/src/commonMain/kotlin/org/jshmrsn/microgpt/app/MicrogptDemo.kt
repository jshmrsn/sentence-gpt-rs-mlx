package org.jshmrsn.microgpt.app

import microgpt_kotlin_visualized.shared.generated.resources.Res
import org.jetbrains.compose.resources.ExperimentalResourceApi
import org.jshmrsn.microgpt.lib.TrainedMicrogpt
import org.jshmrsn.microgpt.lib.generateSamples
import org.jshmrsn.microgpt.lib.microgpt
import kotlin.random.Random

private const val GeneratedExpressionCount = 50_000
private const val MaximumOperand = 100

fun generateMicrogptInputText(expressionCount: Int = GeneratedExpressionCount): String =
    buildString {
        val operandCount = MaximumOperand + 1
        repeat(expressionCount) { expressionIndex ->
            val a = expressionIndex % operandCount
            val b = (expressionIndex / operandCount) % operandCount
            append(a)
            append('+')
            append(b)
            append('=')
            append(a + b)
            append('\n')
        }
    }

@OptIn(ExperimentalResourceApi::class)
suspend fun loadMicrogptInputText(): String =
    Res.readBytes("files/input.txt").decodeToString()


suspend fun trainMicrogptDemo(randomNumberGenerator: Random = Random(0)): TrainedMicrogpt {
    //val inputText = generateMicrogptInputText()
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
