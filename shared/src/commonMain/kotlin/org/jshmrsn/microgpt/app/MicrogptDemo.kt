package org.jshmrsn.microgpt.app

import org.jshmrsn.microgpt.lib.MicrogptTrainingProgress
import org.jshmrsn.microgpt.lib.MicrogptTrainingSession
import org.jshmrsn.microgpt.lib.TrainedMicrogpt
import org.jshmrsn.microgpt.lib.createMicrogptTrainingSession
import org.jshmrsn.microgpt.lib.generateSamples
import org.jshmrsn.microgpt.lib.trainMicrogptStep
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

fun createMicrogptDemoTrainingSession(randomNumberGenerator: Random = Random(0)): MicrogptTrainingSession =
    createMicrogptTrainingSession(
        inputText = generateMicrogptInputText(),
        randomNumberGenerator = randomNumberGenerator
    )

fun trainMicrogptDemoStep(session: MicrogptTrainingSession): MicrogptTrainingProgress? =
    trainMicrogptStep(session)

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
