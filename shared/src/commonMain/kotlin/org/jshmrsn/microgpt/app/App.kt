package org.jshmrsn.microgpt.app

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import org.jshmrsn.microgpt.lib.TrainedMicrogpt
import kotlin.math.roundToLong
import kotlin.random.Random

private fun formatLoss(loss: Double): String {
    val scaled = (loss * 10_000.0).roundToLong()
    val whole = scaled / 10_000
    val fraction = (scaled % 10_000).toString().padStart(4, '0')
    return "$whole.$fraction"
}

@Composable
@Preview
fun App() {
    MaterialTheme {
        var trainedMicrogpt by remember { mutableStateOf<TrainedMicrogpt?>(null) }
        var completedStepCount by remember { mutableStateOf(0) }
        var trainingStepCount by remember { mutableStateOf(1) }
        var latestLoss by remember { mutableStateOf<Double?>(null) }
        var prefix by remember { mutableStateOf("1+3=") }
        var samples by remember { mutableStateOf(emptyList<String>()) }
        val sampleRandomNumberGenerator = remember { Random(1) }

        LaunchedEffect(Unit) {
            var trainingSession = createMicrogptDemoTrainingSession()
            trainingStepCount = trainingSession.trainingStepCount

            while (!trainingSession.isComplete) {
                val result = trainMicrogptDemoStep(trainingSession) ?: break
                trainingSession = result.session
                val progress = result.progress
                completedStepCount = progress.completedStepCount
                trainingStepCount = progress.trainingStepCount
                latestLoss = progress.loss
                withFrameNanos { }
            }

            trainedMicrogpt = trainingSession.trainedMicrogpt
        }

        Column(
            modifier = Modifier
                .background(MaterialTheme.colorScheme.primaryContainer)
                .safeContentPadding()
                .fillMaxSize()
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
            latestLoss?.let { loss ->
                Text("Loss ${formatLoss(loss)}")
            }
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
                        randomNumberGenerator = sampleRandomNumberGenerator,
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
