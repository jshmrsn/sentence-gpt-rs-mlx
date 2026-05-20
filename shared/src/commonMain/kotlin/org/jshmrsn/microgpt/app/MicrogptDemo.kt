package org.jshmrsn.microgpt.app

import org.jshmrsn.microgpt.lib.MicrogptTrainingSession
import org.jshmrsn.microgpt.lib.MicrogptTrainingStepResult
import org.jshmrsn.microgpt.lib.TrainedMicrogpt
import org.jshmrsn.microgpt.lib.generateSamples
import org.jshmrsn.microgpt.lib.trainMicrogptStep
import kotlin.random.Random

private const val MaximumOperand = 999

fun generateMathTrainingText(): List<String> =
    mutableListOf<String>().also {
        for (a in 0..MaximumOperand) {
            for (b in 0..MaximumOperand) {
                it.add(buildString {
                    append(a)
                    append('+')
                    append(b)
                    append('=')
                    append(a + b)
                })
            }
        }
    }

fun trainMicrogptDemoStep(session: MicrogptTrainingSession): MicrogptTrainingStepResult? =
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
