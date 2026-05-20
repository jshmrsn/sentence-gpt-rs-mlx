package org.jshmrsn.microgpt.app

import org.jshmrsn.microgpt.lib.MicrogptTrainingSession
import org.jshmrsn.microgpt.lib.MicrogptTrainingStepResult
import org.jshmrsn.microgpt.lib.TrainedMicrogpt
import org.jshmrsn.microgpt.lib.generateSamples
import org.jshmrsn.microgpt.lib.trainMicrogptStep
import kotlin.random.Random


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
