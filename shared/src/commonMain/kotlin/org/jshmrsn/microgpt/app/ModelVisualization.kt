package org.jshmrsn.microgpt.app

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RichTooltip
import androidx.compose.material3.Text
import androidx.compose.material3.TooltipAnchorPosition
import androidx.compose.material3.TooltipBox
import androidx.compose.material3.TooltipDefaults
import androidx.compose.material3.rememberTooltipState
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.Layout
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import org.jshmrsn.microgpt.lib.Matrix
import org.jshmrsn.microgpt.lib.TrainedMicrogpt
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

private val VocabLabelWidth = 72.dp
private val NetworkHorizontalPadding = 18.dp
private val NetworkVerticalPadding = 14.dp

@Composable
fun MicrogptModelVisualization(
    trainedMicrogpt: TrainedMicrogpt?,
    completedStepCount: Int,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            text = "Model values",
            style = MaterialTheme.typography.titleMedium
        )

        if (trainedMicrogpt == null) {
            Text("Initializing model")
            return@Column
        }

        Text(
            text = "Step $completedStepCount | layers ${trainedMicrogpt.config.layerCount} | embedding ${trainedMicrogpt.config.embeddingSize} | heads ${trainedMicrogpt.config.attentionHeadCount} x ${trainedMicrogpt.config.attentionHeadSize} | context ${trainedMicrogpt.config.contextWindowSize} | vocab ${trainedMicrogpt.tokenizer.vocabularySize}",
            style = MaterialTheme.typography.labelMedium
        )
        NetworkDiagram(
            trainedMicrogpt = trainedMicrogpt,
            modifier = Modifier
                .fillMaxWidth()
                .height(1400.dp)
        )
        EmbeddingAndHeadHeatmaps(trainedMicrogpt)
        TransformerLayerVisualizations(trainedMicrogpt)
    }
}

@Composable
private fun NetworkDiagram(
    trainedMicrogpt: TrainedMicrogpt,
    modifier: Modifier = Modifier
) {
    val stages = buildArchitectureStages(trainedMicrogpt)
    val edgeMatrices = stages.drop(1).mapNotNull { it.incomingWeights }

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(modifier = Modifier.fillMaxWidth()) {
            Spacer(Modifier.width(VocabLabelWidth))
            StageLabels(
                stages = stages,
                modifier = Modifier
                    .weight(1f)
                    .height(56.dp)
            )
            Spacer(Modifier.width(VocabLabelWidth))
        }
        Row(modifier = modifier) {
            VocabRowLabels(
                trainedMicrogpt = trainedMicrogpt,
                modifier = Modifier
                    .width(VocabLabelWidth)
                    .fillMaxHeight()
            )
            Canvas(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
            ) {
                val stageCount = stages.size
                val horizontalPadding = NetworkHorizontalPadding.toPx()
                val verticalPadding = NetworkVerticalPadding.toPx()
                val xStep = (size.width - horizontalPadding * 2f) / (stageCount - 1).toFloat()
                val maxNodeCount = stages.maxOf { it.nodeCount }
                val nodeSpacing = if (maxNodeCount <= 1) {
                    size.height - verticalPadding * 2f
                } else {
                    (size.height - verticalPadding * 2f) / (maxNodeCount - 1).toFloat()
                }
                val radius = min(5.dp.toPx(), max(0.75.dp.toPx(), nodeSpacing * 0.45f))
                val maxMagnitude = max(0.001, edgeMatrices.maxOf { matrixMaxAbs(it) })

                for (stageIndex in 0 until stageCount - 1) {
                    val matrix = stages[stageIndex + 1].incomingWeights ?: continue
                    val fromNodeIndices = sampledNodeIndices(stages[stageIndex].nodeCount)
                    val toNodeIndices = sampledNodeIndices(stages[stageIndex + 1].nodeCount)
                    for (fromIndex in fromNodeIndices) {
                        for (toIndex in toNodeIndices) {
                            val value = connectionValue(
                                matrix = matrix,
                                fromStage = stages[stageIndex],
                                toStage = stages[stageIndex + 1],
                                fromNodeIndex = fromIndex,
                                toNodeIndex = toIndex
                            )
                            val strength = min(1f, (abs(value) / maxMagnitude).toFloat())
                            val color = weightColor(value).copy(alpha = 0.15f + strength * 0.65f)
                            val start = Offset(
                                x = horizontalPadding + xStep * stageIndex,
                                y = nodeY(fromIndex, stages[stageIndex], size.height, verticalPadding)
                            )
                            val end = Offset(
                                x = horizontalPadding + xStep * (stageIndex + 1),
                                y = nodeY(toIndex, stages[stageIndex + 1], size.height, verticalPadding)
                            )
                            drawLine(
                                color = color,
                                start = start,
                                end = end,
                                strokeWidth = 0.5.dp.toPx() + strength * 2.5.dp.toPx(),
                                cap = StrokeCap.Round
                            )
                        }
                    }
                }

                for (stageIndex in 0 until stageCount) {
                    for (nodeIndex in 0 until stages[stageIndex].nodeCount) {
                        val center = Offset(
                            x = horizontalPadding + xStep * stageIndex,
                            y = nodeY(nodeIndex, stages[stageIndex], size.height, verticalPadding)
                        )
                        drawCircle(Color(0xFFF7F9FB), radius = radius, center = center)
                        drawCircle(
                            color = Color(0xFF2F3A42),
                            radius = radius,
                            center = center,
                            style = Stroke(width = 1.dp.toPx())
                        )
                    }
                }
            }
            VocabRowLabels(
                trainedMicrogpt = trainedMicrogpt,
                modifier = Modifier
                    .width(VocabLabelWidth)
                    .fillMaxHeight()
            )
        }
    }
}

@Composable
private fun StageLabels(
    stages: List<ArchitectureStage>,
    modifier: Modifier = Modifier,
    horizontalPadding: Dp = NetworkHorizontalPadding
) {
    val textStyle = MaterialTheme.typography.labelSmall
    Layout(
        modifier = modifier,
        content = {
            stages.forEach { stage ->
                StageLabel(stage = stage, textStyle = textStyle)
            }
        }
    ) { measurables, constraints ->
        val horizontalPaddingPx = horizontalPadding.roundToPx()
        val availableWidth = (constraints.maxWidth - horizontalPaddingPx * 2).coerceAtLeast(1)
        val xStep = if (stages.size > 1) {
            availableWidth.toFloat() / (stages.size - 1).toFloat()
        } else {
            0f
        }
        val maxLabelWidth = if (stages.size > 1) {
            xStep.roundToInt().coerceAtLeast(24.dp.roundToPx())
        } else {
            constraints.maxWidth
        }
        val placeables = measurables.map { measurable ->
            measurable.measure(
                constraints.copy(
                    minWidth = 0,
                    minHeight = 0,
                    maxWidth = maxLabelWidth
                )
            )
        }
        val layoutHeight = placeables
            .maxOfOrNull { it.height }
            ?.coerceIn(constraints.minHeight, constraints.maxHeight)
            ?: constraints.minHeight

        layout(constraints.maxWidth, layoutHeight) {
            placeables.forEachIndexed { stageIndex, placeable ->
                val columnCenterX = horizontalPaddingPx + xStep * stageIndex
                val x = (columnCenterX - placeable.width / 2f)
                    .roundToInt()
                    .coerceIn(0, constraints.maxWidth - placeable.width)
                placeable.placeRelative(x, 0)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun StageLabel(
    stage: ArchitectureStage,
    textStyle: TextStyle
) {
    TooltipBox(
        positionProvider = TooltipDefaults.rememberTooltipPositionProvider(TooltipAnchorPosition.Above),
        state = rememberTooltipState(),
        tooltip = {
            RichTooltip(
                title = { Text(stage.fullName) },
                text = { Text(stage.explanation) }
            )
        }
    ) {
        Text(
            text = stage.label,
            style = textStyle,
            textAlign = TextAlign.Center,
            maxLines = 3
        )
    }
}

@Composable
private fun VocabRowLabels(
    trainedMicrogpt: TrainedMicrogpt,
    modifier: Modifier = Modifier
) {
    val labels = tokenLabels(trainedMicrogpt)
    val textStyle = MaterialTheme.typography.labelSmall
    Layout(
        modifier = modifier,
        content = {
            labels.forEach { label ->
                Text(
                    text = label,
                    style = textStyle,
                    maxLines = 1
                )
            }
        }
    ) { measurables, constraints ->
        val verticalPaddingPx = NetworkVerticalPadding.roundToPx()
        val placeables = measurables.map { measurable ->
            measurable.measure(
                constraints.copy(
                    minWidth = 0,
                    minHeight = 0,
                    maxWidth = constraints.maxWidth
                )
            )
        }
        layout(constraints.maxWidth, constraints.maxHeight) {
            placeables.forEachIndexed { tokenIndex, placeable ->
                val y = (featureY(
                    featureIndex = tokenIndex,
                    featureCount = labels.size,
                    height = constraints.maxHeight.toFloat(),
                    verticalPadding = verticalPaddingPx.toFloat()
                ) - placeable.height / 2f)
                    .roundToInt()
                    .coerceIn(0, constraints.maxHeight - placeable.height)
                placeable.placeRelative(0, y)
            }
        }
    }
}

@Composable
private fun EmbeddingAndHeadHeatmaps(trainedMicrogpt: TrainedMicrogpt) {
    val matrices = listOf(
        "Token embedding ${matrixShape(trainedMicrogpt.model.tokenEmbedding)}" to trainedMicrogpt.model.tokenEmbedding,
        "Position embedding ${matrixShape(trainedMicrogpt.model.positionEmbedding)}" to trainedMicrogpt.model.positionEmbedding,
        "Language head ${matrixShape(trainedMicrogpt.model.languageModelHead)}" to trainedMicrogpt.model.languageModelHead
    )
    val globalScale = max(0.001, matrices.maxOf { (_, matrix) -> matrixMaxAbs(matrix) })

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("Embeddings and output", style = MaterialTheme.typography.titleSmall)
        matrices.chunked(2).forEach { rowMatrices ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                rowMatrices.forEach { (label, matrix) ->
                    MatrixHeatmap(
                        label = label,
                        matrix = matrix,
                        scale = globalScale,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }
    }
}

@Composable
private fun TransformerLayerVisualizations(trainedMicrogpt: TrainedMicrogpt) {
    val configuredLayerCount = trainedMicrogpt.config.layerCount
    val actualLayerCount = trainedMicrogpt.model.layers.size
    Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Text(
            text = "Transformer layers $actualLayerCount / configured $configuredLayerCount",
            style = MaterialTheme.typography.titleSmall
        )
        trainedMicrogpt.model.layers.forEachIndexed { layerIndex, layer ->
            val attention = layer.attention
            val feedForward = layer.feedForward
            val matrices = listOf(
                "Q ${matrixShape(attention.queryWeights)}" to attention.queryWeights,
                "K ${matrixShape(attention.keyWeights)}" to attention.keyWeights,
                "V ${matrixShape(attention.valueWeights)}" to attention.valueWeights,
                "Attn out ${matrixShape(attention.outputProjectionWeights)}" to attention.outputProjectionWeights,
                "FF expand ${matrixShape(feedForward.expansionWeights)}" to feedForward.expansionWeights,
                "FF project ${matrixShape(feedForward.projectionWeights)}" to feedForward.projectionWeights
            )
            val globalScale = max(0.001, matrices.maxOf { (_, matrix) -> matrixMaxAbs(matrix) })

            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    text = "Layer ${layerIndex + 1}",
                    style = MaterialTheme.typography.labelLarge
                )
                matrices.chunked(2).forEach { rowMatrices ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        rowMatrices.forEach { (label, matrix) ->
                            MatrixHeatmap(
                                label = label,
                                matrix = matrix,
                                scale = globalScale,
                                modifier = Modifier.weight(1f)
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun MatrixHeatmap(
    label: String,
    matrix: Matrix,
    scale: Double,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(label, style = MaterialTheme.typography.labelMedium)
        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(120.dp)
        ) {
            val rows = matrix.size
            val columns = matrix.firstOrNull()?.size ?: 0
            if (rows == 0 || columns == 0) return@Canvas

            val cellWidth = size.width / columns.toFloat()
            val cellHeight = size.height / rows.toFloat()
            for (rowIndex in 0 until rows) {
                for (columnIndex in 0 until columns) {
                    val value = matrix[rowIndex][columnIndex].data
                    val strength = min(1f, (abs(value) / scale).toFloat())
                    drawRect(
                        color = weightColor(value).copy(alpha = 0.12f + strength * 0.88f),
                        topLeft = Offset(columnIndex * cellWidth, rowIndex * cellHeight),
                        size = Size(cellWidth, cellHeight)
                    )
                }
            }
        }
        Text(matrixStats(matrix), style = MaterialTheme.typography.labelSmall)
    }
}

private fun matrixValueOrZero(matrix: Matrix, rowIndex: Int, columnIndex: Int): Double =
    matrix.getOrNull(rowIndex)?.getOrNull(columnIndex)?.data ?: 0.0

private fun matrixMaxAbs(matrix: Matrix): Double =
    matrix.maxOfOrNull { row -> row.maxOfOrNull { value -> abs(value.data) } ?: 0.0 } ?: 0.0

private fun matrixStats(matrix: Matrix): String {
    val values = matrix.flatMap { row -> row.map { it.data } }
    if (values.isEmpty()) return "empty"
    val minValue = values.minOrNull() ?: 0.0
    val maxValue = values.maxOrNull() ?: 0.0
    val meanAbs = values.sumOf { abs(it) } / values.size.toDouble()
    return "min ${formatCompact(minValue)} max ${formatCompact(maxValue)} mean |w| ${formatCompact(meanAbs)}"
}

private data class ArchitectureStage(
    val label: String,
    val fullName: String,
    val explanation: String,
    val nodeCount: Int,
    val featureCount: Int,
    val sequenceLength: Int,
    val incomingWeights: Matrix?,
    val incomingWeightKind: IncomingWeightKind = IncomingWeightKind.Linear
)

private enum class IncomingWeightKind {
    Linear,
    TokenEmbedding
}

private fun buildArchitectureStages(trainedMicrogpt: TrainedMicrogpt): List<ArchitectureStage> {
    val contextWindowSize = trainedMicrogpt.config.contextWindowSize
    val vocabularySize = trainedMicrogpt.tokenizer.vocabularySize
    val embeddingSize = trainedMicrogpt.config.embeddingSize
    val feedForwardSize = embeddingSize * 4

    fun stage(
        label: String,
        fullName: String,
        explanation: String,
        featureCount: Int,
        incomingWeights: Matrix?,
        incomingWeightKind: IncomingWeightKind = IncomingWeightKind.Linear,
        sequenceLength: Int = contextWindowSize
    ): ArchitectureStage =
        ArchitectureStage(
            label = if (sequenceLength == 1) "$label\n$featureCount" else "$label\n${sequenceLength}x$featureCount",
            fullName = fullName,
            explanation = explanation,
            nodeCount = sequenceLength * featureCount,
            featureCount = featureCount,
            sequenceLength = sequenceLength,
            incomingWeights = incomingWeights,
            incomingWeightKind = incomingWeightKind
        )

    return listOf(
        stage(
            label = "Token",
            fullName = "Input token one-hot activations",
            explanation = "Each context position is represented as a one-hot vector over the vocabulary. Rows are grouped by token id and repeated across context positions.",
            featureCount = vocabularySize,
            incomingWeights = null
        ),
        stage(
            label = "Embed",
            fullName = "Token embedding",
            explanation = "Maps token ids into learned dense vectors. This is the token embedding table applied at each context position.",
            featureCount = embeddingSize,
            incomingWeights = trainedMicrogpt.model.tokenEmbedding,
            incomingWeightKind = IncomingWeightKind.TokenEmbedding
        )
    ) +
        trainedMicrogpt.model.layers.flatMapIndexed { layerIndex, layer ->
            listOf(
                stage(
                    label = "L${layerIndex + 1} QKV",
                    fullName = "Layer ${layerIndex + 1} query/key/value projections",
                    explanation = "The hidden state is projected into attention query, key, and value spaces. The diagram samples the query matrix for this stage.",
                    featureCount = embeddingSize,
                    incomingWeights = layer.attention.queryWeights
                ),
                stage(
                    label = "L${layerIndex + 1} Attn",
                    fullName = "Layer ${layerIndex + 1} attention output projection",
                    explanation = "Attention mixes information across earlier context positions, then projects the concatenated heads back to embedding width.",
                    featureCount = embeddingSize,
                    incomingWeights = layer.attention.outputProjectionWeights
                ),
                stage(
                    label = "L${layerIndex + 1} FF+",
                    fullName = "Layer ${layerIndex + 1} feed-forward expansion",
                    explanation = "The position-wise feed-forward network expands each hidden vector to four times the embedding width before ReLU.",
                    featureCount = feedForwardSize,
                    incomingWeights = layer.feedForward.expansionWeights
                ),
                stage(
                    label = "L${layerIndex + 1} FF-",
                    fullName = "Layer ${layerIndex + 1} feed-forward projection",
                    explanation = "The feed-forward network projects the expanded activations back down to embedding width before the residual update.",
                    featureCount = embeddingSize,
                    incomingWeights = layer.feedForward.projectionWeights
                )
            )
        } +
        listOf(
            stage(
                label = "Logits",
                fullName = "Vocabulary logits",
                explanation = "This implementation runs the transformer one position at a time. The language-model head returns one score per vocabulary token for the current position.",
                featureCount = vocabularySize,
                incomingWeights = trainedMicrogpt.model.languageModelHead,
                sequenceLength = 1
            )
        )
}

private fun connectionValue(
    matrix: Matrix,
    fromStage: ArchitectureStage,
    toStage: ArchitectureStage,
    fromNodeIndex: Int,
    toNodeIndex: Int
): Double {
    val fromFeatureIndex = fromNodeIndex % fromStage.featureCount
    val toFeatureIndex = toNodeIndex % toStage.featureCount
    return when (toStage.incomingWeightKind) {
        IncomingWeightKind.Linear -> matrixValueOrZero(matrix, toFeatureIndex, fromFeatureIndex)
        IncomingWeightKind.TokenEmbedding -> matrixValueOrZero(matrix, fromFeatureIndex, toFeatureIndex)
    }
}

private fun nodeY(
    nodeIndex: Int,
    stage: ArchitectureStage,
    height: Float,
    verticalPadding: Float
): Float {
    val featureIndex = nodeIndex % stage.featureCount
    val positionIndex = nodeIndex / stage.featureCount
    val baseY = featureY(featureIndex, stage.featureCount, height, verticalPadding)
    if (stage.sequenceLength <= 1) return baseY

    val featureSpacing = if (stage.featureCount <= 1) {
        height - verticalPadding * 2f
    } else {
        (height - verticalPadding * 2f) / (stage.featureCount - 1).toFloat()
    }
    val spread = min(featureSpacing * 0.72f, 22f)
    val positionOffset = -spread / 2f +
        spread * positionIndex.toFloat() / (stage.sequenceLength - 1).toFloat()
    return baseY + positionOffset
}

private fun featureY(
    featureIndex: Int,
    featureCount: Int,
    height: Float,
    verticalPadding: Float
): Float {
    if (featureCount <= 1) return height / 2f
    val yStep = (height - verticalPadding * 2f) / (featureCount - 1).toFloat()
    return verticalPadding + yStep * featureIndex
}

private fun sampledNodeIndices(nodeCount: Int, maxSampledNodeCount: Int = 8): List<Int> {
    if (nodeCount <= maxSampledNodeCount) return (0 until nodeCount).toList()
    val lastIndex = nodeCount - 1
    return (0 until maxSampledNodeCount)
        .map { sampleIndex ->
            (sampleIndex.toDouble() * lastIndex.toDouble() / (maxSampledNodeCount - 1).toDouble())
                .roundToInt()
        }
        .distinct()
}

private fun tokenLabels(trainedMicrogpt: TrainedMicrogpt): List<String> =
    (0 until trainedMicrogpt.tokenizer.vocabularySize).map { tokenIndex ->
        "$tokenIndex ${tokenDisplay(trainedMicrogpt, tokenIndex)}"
    }

private fun tokenDisplay(trainedMicrogpt: TrainedMicrogpt, tokenIndex: Int): String =
    if (tokenIndex == trainedMicrogpt.tokenizer.sequenceBoundaryTokenId) {
        "<B>"
    } else {
        val character = trainedMicrogpt.tokenizer.uniqueCharacters[tokenIndex]
        when (character) {
            '\n' -> "\\n"
            '\t' -> "\\t"
            ' ' -> "' '"
            else -> character.toString()
        }
    }

private fun matrixShape(matrix: Matrix): String {
    val rows = matrix.size
    val columns = matrix.firstOrNull()?.size ?: 0
    return "${rows}x$columns"
}

private fun weightColor(value: Double): Color =
    if (value >= 0.0) Color(0xFF1976D2) else Color(0xFFC62828)

private fun formatCompact(value: Double): String {
    val scaled = (value * 1000.0).toInt()
    val sign = if (scaled < 0) "-" else ""
    val absolute = abs(scaled)
    return "$sign${absolute / 1000}.${(absolute % 1000).toString().padStart(3, '0')}"
}
