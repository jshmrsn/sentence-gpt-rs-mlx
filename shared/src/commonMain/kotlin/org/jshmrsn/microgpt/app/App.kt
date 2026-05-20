package org.jshmrsn.microgpt.app

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import org.jshmrsn.microgpt.lib.TrainedMicrogpt
import kotlin.random.Random

@Composable
@Preview
fun App() {
    MaterialTheme {
        var trainedMicrogpt by remember { mutableStateOf<TrainedMicrogpt?>(null) }
        var microgptStatus by remember { mutableStateOf("Training...") }
        var prefix by remember { mutableStateOf("") }
        var samples by remember { mutableStateOf(emptyList<String>()) }
        val sampleRandomNumberGenerator = remember { Random(1) }

        LaunchedEffect(Unit) {
            trainedMicrogpt = trainMicrogptDemo()
            microgptStatus = "Ready"
        }

        Column(
            modifier = Modifier
                .background(MaterialTheme.colorScheme.primaryContainer)
                .safeContentPadding()
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(microgptStatus)
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
