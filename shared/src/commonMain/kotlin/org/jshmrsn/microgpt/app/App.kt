package org.jshmrsn.microgpt.app

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import microgpt_kotlin_visualized.shared.generated.resources.Res
import org.jshmrsn.microgpt.lib.MicrogptTrainingProgress
import org.jshmrsn.microgpt.lib.MicrogptTrainingSession
import org.jshmrsn.microgpt.lib.MicrogptTrainingStepResult
import org.jshmrsn.microgpt.lib.TrainedMicrogpt
import org.jshmrsn.microgpt.lib.calculateDocumentLoss
import org.jshmrsn.microgpt.lib.createMicrogptTrainingSession
import kotlin.math.ln
import kotlin.math.roundToLong
import kotlin.random.Random
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.TimeSource

private val TrainingFrameBudget = 100.milliseconds
private const val ValidationStepInterval = 300

private fun formatLoss(loss: Double): String {
    val scaled = (loss * 10_000.0).roundToLong()
    val whole = scaled / 10_000
    val fraction = (scaled % 10_000).toString().padStart(4, '0')
    return "$whole.$fraction"
}

private fun formatPercent(value: Double): String {
    val scaled = (value * 1000.0).roundToLong()
    val whole = scaled / 10
    val fraction = (scaled % 10).toString().padStart(1, '0')
    return "$whole.$fraction%"
}

private fun estimatedAccuracyFromLoss(loss: Double, vocabularySize: Int): Double {
    val randomLoss = ln(vocabularySize.toDouble())
    if (randomLoss <= 0.0) return 1.0
    return (1.0 - loss / randomLoss).coerceIn(0.0, 1.0)
}

private fun calculateValidationLoss(session: MicrogptTrainingSession, completedStepCount: Int): Double? {
    if (session.validationDocuments.isEmpty()) return null

    val validationDocumentCount = minOf(
        session.validationEvaluationDocumentCount,
        session.validationDocuments.size
    )
    val validationBatchIndex = completedStepCount / ValidationStepInterval
    return (0 until validationDocumentCount).sumOf { validationOffset ->
        val validationIndex =
            (validationBatchIndex * validationDocumentCount + validationOffset) % session.validationDocuments.size
        val validationDocument = session.validationDocuments[validationIndex]
        calculateDocumentLoss(
            model = session.trainedMicrogpt.model,
            config = session.trainedMicrogpt.config,
            tokenizer = session.trainedMicrogpt.tokenizer,
            document = validationDocument
        )
    } / validationDocumentCount.toDouble()
}

private fun calculateTrainingLossBaseline(session: MicrogptTrainingSession): Double? {
    val document = session.documents.firstOrNull() ?: return null
    return calculateDocumentLoss(
        model = session.trainedMicrogpt.model,
        config = session.trainedMicrogpt.config,
        tokenizer = session.trainedMicrogpt.tokenizer,
        document = document
    )
}

private fun MicrogptTrainingSession.withInitialProgress(
    trainLoss: Double?,
    validationLoss: Double?
): MicrogptTrainingSession {
    if (trainLoss == null && validationLoss == null) return this

    val progress = MicrogptTrainingProgress(
        completedStepCount = 0,
        trainingStepCount = trainingStepCount,
        loss = trainLoss ?: validationLoss ?: 0.0,
        validationLoss = validationLoss
    )
    return copy(
        latestLoss = trainLoss,
        latestValidationLoss = validationLoss,
        progressHistory = listOf(progress)
    )
}

private fun MicrogptTrainingStepResult.withValidationLoss(
    validationLoss: Double?
): MicrogptTrainingStepResult {
    if (validationLoss == null) return this

    val progressWithValidation = progress.copy(validationLoss = validationLoss)
    val progressHistoryWithValidation = progressHistory.dropLast(1) + progressWithValidation
    val sessionWithValidation = session.copy(
        latestValidationLoss = validationLoss,
        progressHistory = progressHistoryWithValidation
    )

    return copy(
        session = sessionWithValidation,
        progress = progressWithValidation,
        progressHistory = progressHistoryWithValidation
    )
}

@Composable
@Preview
fun App() {
    MaterialTheme {
        var trainedMicrogpt by remember { mutableStateOf<TrainedMicrogpt?>(null) }
        var completedStepCount by remember { mutableStateOf(0) }
        var trainingStepCount by remember { mutableStateOf(1) }
        var latestLoss by remember { mutableStateOf<Double?>(null) }
        var latestValidationLoss by remember { mutableStateOf<Double?>(null) }
        var progressHistory by remember { mutableStateOf(emptyList<MicrogptTrainingProgress>()) }
        var trainingExampleCount by remember { mutableStateOf(0) }
        var validationExampleCount by remember { mutableStateOf(0) }
        var validationEvaluationExampleCount by remember { mutableStateOf(0) }
        var prefix by remember { mutableStateOf("1+3=") }
        var samples by remember { mutableStateOf(emptyList<String>()) }
        var visibleMicrogpt by remember { mutableStateOf<TrainedMicrogpt?>(null) }
        val randomNumberGenerator = remember { Random(1) }

        LaunchedEffect(Unit) {
            val trainingText = if (true) {
                generateMathTrainingText()
            } else {
                Res.readBytes("files/input.txt").decodeToString()
            }

            var trainingSession = createMicrogptTrainingSession(
                inputText = trainingText,
                randomNumberGenerator = randomNumberGenerator,
                trainingStepCount = 10000,
                validationDivisor = 20
            )

            trainingStepCount = trainingSession.trainingStepCount
            trainingExampleCount = trainingSession.documents.size
            validationExampleCount = trainingSession.validationDocuments.size
            validationEvaluationExampleCount = minOf(
                trainingSession.validationEvaluationDocumentCount,
                trainingSession.validationDocuments.size
            )
            trainingSession = trainingSession.withInitialProgress(
                trainLoss = calculateTrainingLossBaseline(trainingSession),
                validationLoss = calculateValidationLoss(
                    session = trainingSession,
                    completedStepCount = 0
                )
            )
            latestLoss = trainingSession.latestLoss
            latestValidationLoss = trainingSession.latestValidationLoss
            progressHistory = trainingSession.progressHistory
            visibleMicrogpt = trainingSession.trainedMicrogpt
            var nextValidationStep = ValidationStepInterval

            while (!trainingSession.isComplete) {
                val frameStart = TimeSource.Monotonic.markNow()
                var latestResult: MicrogptTrainingStepResult? = null

                do {
                    val result = trainMicrogptDemoStep(trainingSession) ?: break
                    trainingSession = result.session
                    latestResult = result
                } while (
                    !trainingSession.isComplete &&
                    trainingSession.completedStepCount < nextValidationStep &&
                    frameStart.elapsedNow() < TrainingFrameBudget
                )

                var result = latestResult ?: break
                if (trainingSession.completedStepCount >= nextValidationStep) {
                    result = result.withValidationLoss(
                        calculateValidationLoss(
                            session = trainingSession,
                            completedStepCount = trainingSession.completedStepCount
                        )
                    )
                    trainingSession = result.session
                    nextValidationStep += ValidationStepInterval
                }
                val progress = result.progress
                completedStepCount = progress.completedStepCount
                trainingStepCount = progress.trainingStepCount
                latestLoss = progress.loss
                latestValidationLoss = progress.validationLoss ?: trainingSession.latestValidationLoss
                progressHistory = result.progressHistory
                visibleMicrogpt = trainingSession.trainedMicrogpt
                withFrameNanos { }
            }

            trainedMicrogpt = trainingSession.trainedMicrogpt
        }

        Column(
            modifier = Modifier
                .background(MaterialTheme.colorScheme.primaryContainer)
                .safeContentPadding()
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            val isTraining = trainedMicrogpt == null
            val progress = completedStepCount.toFloat() / trainingStepCount.toFloat()
            Text(if (isTraining) "Training" else "Ready")
            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier.fillMaxWidth(),
            )
            Text("Step $completedStepCount / $trainingStepCount")
            if (trainingExampleCount > 0 || validationExampleCount > 0) {
                Text(
                    text = "Train examples $trainingExampleCount | validation examples $validationExampleCount | validation batch $validationEvaluationExampleCount",
                    style = MaterialTheme.typography.labelMedium
                )
            }
            val vocabularySize = visibleMicrogpt?.tokenizer?.vocabularySize ?: 0
            latestLoss?.let { loss ->
                LossMetricText(
                    label = "Train loss",
                    loss = loss,
                    vocabularySize = vocabularySize
                )
            }
            latestValidationLoss?.let { validationLoss ->
                LossMetricText(
                    label = "Validation loss",
                    loss = validationLoss,
                    vocabularySize = vocabularySize
                )
            }
            LossHistoryChart(
                progressHistory = progressHistory,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 8.dp)
            )
            MicrogptModelVisualization(
                trainedMicrogpt = visibleMicrogpt,
                completedStepCount = completedStepCount,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 8.dp)
            )
            OutlinedTextField(
                value = prefix,
                onValueChange = { prefix = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Prefix") },
                singleLine = true,
            )
            Button(
                enabled = trainedMicrogpt != null,
                onClick = {
                    val model = trainedMicrogpt ?: return@Button
                    samples = generateMicrogptSamples(
                        trainedMicrogpt = model,
                        prefix = prefix,
                        randomNumberGenerator = randomNumberGenerator,
                        sampleCount = 10,
                        temperature = 0.5
                    )
                }
            ) {
                Text("Generate")
            }
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                samples.forEachIndexed { index, sample ->
                    Text("Sample ${index + 1}: $sample")
                }
            }
        }
    }
}

@Composable
private fun LossMetricText(
    label: String,
    loss: Double,
    vocabularySize: Int
) {
    Text("$label ${formatLoss(loss)}")
    if (vocabularySize > 1) {
        val estimatedAccuracy = estimatedAccuracyFromLoss(loss, vocabularySize)
        val randomAccuracy = 1.0 / vocabularySize.toDouble()
        Text(
            text = "$label estimated accuracy ${formatPercent(estimatedAccuracy)} | random ${formatPercent(randomAccuracy)} | vocab $vocabularySize",
            style = MaterialTheme.typography.labelMedium
        )
    }
}

@Composable
private fun LossHistoryChart(
    progressHistory: List<MicrogptTrainingProgress>,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text("Loss over steps", style = MaterialTheme.typography.titleSmall)

        if (progressHistory.isEmpty()) {
            Text("Waiting for first training step", style = MaterialTheme.typography.labelMedium)
            return@Column
        }

        val plottedLosses = progressHistory.flatMap { progress ->
            listOfNotNull(progress.loss, progress.validationLoss)
        }
        val minLoss = plottedLosses.minOrNull() ?: 0.0
        val maxLoss = plottedLosses.maxOrNull() ?: 0.0
        val lossRange = (maxLoss - minLoss).takeIf { it > 0.000001 } ?: 1.0
        val axisColor = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.55f)
        val gridColor = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.16f)
        val trainingLineColor = MaterialTheme.colorScheme.primary
        val validationLineColor = Color(0xFFC62828)

        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("max ${formatLoss(maxLoss)}", style = MaterialTheme.typography.labelSmall)
            Text("min ${formatLoss(minLoss)}", style = MaterialTheme.typography.labelSmall)
        }
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("train", color = trainingLineColor, style = MaterialTheme.typography.labelSmall)
            Text("validation", color = validationLineColor, style = MaterialTheme.typography.labelSmall)
        }

        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(180.dp)
        ) {
            val leftPadding = 10.dp.toPx()
            val rightPadding = 10.dp.toPx()
            val topPadding = 10.dp.toPx()
            val bottomPadding = 18.dp.toPx()
            val plotWidth = size.width - leftPadding - rightPadding
            val plotHeight = size.height - topPadding - bottomPadding
            val plotLeft = leftPadding
            val plotRight = leftPadding + plotWidth
            val plotTop = topPadding
            val plotBottom = topPadding + plotHeight

            for (gridIndex in 0..4) {
                val y = plotTop + plotHeight * gridIndex.toFloat() / 4f
                drawLine(
                    color = gridColor,
                    start = Offset(plotLeft, y),
                    end = Offset(plotRight, y),
                    strokeWidth = 1.dp.toPx()
                )
            }

            drawLine(
                color = axisColor,
                start = Offset(plotLeft, plotBottom),
                end = Offset(plotRight, plotBottom),
                strokeWidth = 1.dp.toPx()
            )
            drawLine(
                color = axisColor,
                start = Offset(plotLeft, plotTop),
                end = Offset(plotLeft, plotBottom),
                strokeWidth = 1.dp.toPx()
            )

            fun pointAt(index: Int, loss: Double): Offset {
                val x = if (progressHistory.size == 1) {
                    plotLeft
                } else {
                    plotLeft + plotWidth * index.toFloat() / (progressHistory.lastIndex).toFloat()
                }
                val normalizedLoss = ((loss - minLoss) / lossRange).toFloat()
                val y = plotBottom - plotHeight * normalizedLoss
                return Offset(x, y)
            }

            fun drawLossLine(points: List<Offset>, color: Color) {
                if (points.size == 1) {
                    drawCircle(color = color, radius = 4.dp.toPx(), center = points.first())
                } else {
                    points.zipWithNext().forEach { (start, end) ->
                        drawLine(
                            color = color,
                            start = start,
                            end = end,
                            strokeWidth = 2.dp.toPx(),
                            cap = StrokeCap.Round
                        )
                    }
                }
            }

            val trainingPoints = progressHistory.mapIndexed { index, progress ->
                pointAt(index, progress.loss)
            }
            val validationPoints = progressHistory.mapIndexedNotNull { index, progress ->
                progress.validationLoss?.let { validationLoss -> pointAt(index, validationLoss) }
            }
            drawLossLine(trainingPoints, trainingLineColor)
            drawLossLine(validationPoints, validationLineColor)

            validationPoints.forEach { point ->
                drawCircle(color = validationLineColor, radius = 2.dp.toPx(), center = point)
            }
            if (trainingPoints.size == 1) {
                drawCircle(color = trainingLineColor, radius = 4.dp.toPx(), center = trainingPoints.first())
            }
        }

        val latestProgress = progressHistory.last()
        val latestValidationLoss = progressHistory.asReversed().firstNotNullOfOrNull { it.validationLoss }
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(
                text = "Train ${formatLoss(latestProgress.loss)}",
                color = trainingLineColor,
                style = MaterialTheme.typography.labelSmall
            )
            latestValidationLoss?.let { validationLoss ->
                Text(
                    text = "Validation ${formatLoss(validationLoss)}",
                    color = validationLineColor,
                    style = MaterialTheme.typography.labelSmall
                )
            }
        }
        Text(
            text = "Step ${latestProgress.completedStepCount} / ${latestProgress.trainingStepCount}",
            style = MaterialTheme.typography.labelSmall
        )
    }
}
