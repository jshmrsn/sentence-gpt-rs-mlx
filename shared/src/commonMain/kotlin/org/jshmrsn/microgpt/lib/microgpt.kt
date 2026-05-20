package org.jshmrsn.microgpt.lib

/**
 * The most atomic way to train and run inference for a GPT in pure, dependency-free Kotlin.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 *
 * Translation of the provided Python script, with an extra pass of comments
 * informed by the microgpt blog post:
 *
 * - This file contains the algorithmic essence of a GPT:
 *   dataset, tokenizer, autograd, Transformer, optimizer, training, inference.
 * - Almost everything omitted from production systems is omitted for efficiency,
 *   scale, or convenience, not because it changes the core idea.
 * - The model here is a document completer: given a prefix, predict the next token.
 *   A chatbot is still the same next-token prediction mechanism, just on a prompt
 *   that looks like a conversation transcript.
 */


/**
 * BIG PICTURE
 *
 * This tiny file is deliberately not production-grade.
 * It omits:
 * - tensor libraries / GPUs
 * - batching and parallel time-step processing
 * - subword tokenization
 * - RoPE, grouped-query attention, gated activations, MoE, etc.
 * - mixed precision and distributed training
 * - SFT / RL post-training
 * - inference serving infrastructure
 *
 * But the blog's main thesis is that those additions mostly improve scale,
 * efficiency, and product behavior. The algorithmic skeleton is already here:
 *
 * documents
 * -> tokenize
 * -> embed
 * -> Transformer forward pass
 * -> logits
 * -> cross-entropy loss
 * -> backpropagation
 * -> Adam updates
 * -> autoregressive sampling
 *
 * If you understand this file, you understand the core algorithmic essence
 * of GPT training and inference.
 */
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

typealias Matrix = List<List<Value>>
typealias KeyValueCache = List<List<List<Value>>>

data class AttentionParameters(
    val queryWeights: Matrix,
    val keyWeights: Matrix,
    val valueWeights: Matrix,
    val outputProjectionWeights: Matrix
) {
    fun matrices(): List<Matrix> = listOf(queryWeights, keyWeights, valueWeights, outputProjectionWeights)
}

data class FeedForwardParameters(
    val expansionWeights: Matrix,
    val projectionWeights: Matrix
) {
    fun matrices(): List<Matrix> = listOf(expansionWeights, projectionWeights)
}

data class TransformerLayerParameters(
    val attention: AttentionParameters,
    val feedForward: FeedForwardParameters
) {
    fun matrices(): List<Matrix> = attention.matrices() + feedForward.matrices()
}

data class TransformerModelParameters(
    val tokenEmbedding: Matrix,
    val positionEmbedding: Matrix,
    val languageModelHead: Matrix,
    val layers: List<TransformerLayerParameters>
) {
    fun matrices(): List<Matrix> =
        listOf(tokenEmbedding, positionEmbedding, languageModelHead) + layers.flatMap { it.matrices() }

    fun values(): List<Value> =
        matrices().flatMap { matrix -> matrix.flatMap { row -> row } }

    companion object {
        fun initialize(
            vocabularySize: Int,
            contextWindowSize: Int,
            embeddingSize: Int,
            layerCount: Int,
            randomNumberGenerator: Random
        ): TransformerModelParameters =
            TransformerModelParameters(
                tokenEmbedding = matrix(vocabularySize, embeddingSize, randomNumberGenerator),
                positionEmbedding = matrix(contextWindowSize, embeddingSize, randomNumberGenerator),
                languageModelHead = matrix(vocabularySize, embeddingSize, randomNumberGenerator),
                layers = List(layerCount) {
                    TransformerLayerParameters(
                        attention = AttentionParameters(
                            queryWeights = matrix(embeddingSize, embeddingSize, randomNumberGenerator),
                            keyWeights = matrix(embeddingSize, embeddingSize, randomNumberGenerator),
                            valueWeights = matrix(embeddingSize, embeddingSize, randomNumberGenerator),
                            outputProjectionWeights = matrix(embeddingSize, embeddingSize, randomNumberGenerator)
                        ),
                        feedForward = FeedForwardParameters(
                            expansionWeights = matrix(4 * embeddingSize, embeddingSize, randomNumberGenerator),
                            projectionWeights = matrix(embeddingSize, 4 * embeddingSize, randomNumberGenerator)
                        )
                    )
                }
            )
    }
}

private fun TransformerModelParameters.withValues(values: List<Value>): TransformerModelParameters {
    var valueIndex = 0

    fun nextMatrix(matrix: Matrix): Matrix =
        matrix.map { row ->
            row.map {
                values[valueIndex].also { valueIndex += 1 }
            }
        }

    return copy(
        tokenEmbedding = nextMatrix(tokenEmbedding),
        positionEmbedding = nextMatrix(positionEmbedding),
        languageModelHead = nextMatrix(languageModelHead),
        layers = layers.map { layer ->
            layer.copy(
                attention = layer.attention.copy(
                    queryWeights = nextMatrix(layer.attention.queryWeights),
                    keyWeights = nextMatrix(layer.attention.keyWeights),
                    valueWeights = nextMatrix(layer.attention.valueWeights),
                    outputProjectionWeights = nextMatrix(layer.attention.outputProjectionWeights)
                ),
                feedForward = layer.feedForward.copy(
                    expansionWeights = nextMatrix(layer.feedForward.expansionWeights),
                    projectionWeights = nextMatrix(layer.feedForward.projectionWeights)
                )
            )
        }
    )
}

private fun <T> List<T>.replacedAt(index: Int, value: T): List<T> =
    mapIndexed { currentIndex, currentValue -> if (currentIndex == index) value else currentValue }

data class TransformerRun(
    val logits: List<Value>,
    val keys: KeyValueCache,
    val values: KeyValueCache
)

data class TransformerLayerRun(
    val hiddenState: List<Value>,
    val keys: KeyValueCache,
    val values: KeyValueCache
)

data class TransformerConfig(
    val layerCount: Int,
    val embeddingSize: Int,
    val contextWindowSize: Int,
    val attentionHeadCount: Int
) {
    val attentionHeadSize: Int = embeddingSize / attentionHeadCount
}

data class CharacterTokenizer(
    val uniqueCharacters: List<Char>,
    val sequenceBoundaryTokenId: Int
) {
    val vocabularySize: Int = uniqueCharacters.size + 1
    val characterToTokenId: Map<Char, Int> = uniqueCharacters.withIndex().associate { it.value to it.index }

    fun encodeDocument(document: String): List<Int> =
        listOf(sequenceBoundaryTokenId) +
            document.map { character -> characterToTokenId[character]!! } +
            listOf(sequenceBoundaryTokenId)
}

data class AdamOptimizerState(
    val firstMomentEstimates: List<Double>,
    val secondMomentEstimates: List<Double>
)

data class AdamOptimizerConfig(
    val learningRate: Double = 0.01,
    val firstMomentDecay: Double = 0.85,
    val secondMomentDecay: Double = 0.99,
    val epsilon: Double = 1e-8
)

data class TrainedMicrogpt(
    val model: TransformerModelParameters,
    val config: TransformerConfig,
    val tokenizer: CharacterTokenizer
)

data class MicrogptTrainingProgress(
    val completedStepCount: Int,
    val trainingStepCount: Int,
    val loss: Double,
    val validationLoss: Double?
) {
    val isComplete: Boolean = completedStepCount >= trainingStepCount
}

data class MicrogptTrainingSession(
    val trainedMicrogpt: TrainedMicrogpt,
    val documents: List<String>,
    val validationDocuments: List<String>,
    val trainingStepCount: Int,
    val validationEvaluationDocumentCount: Int = 32,
    val optimizerConfig: AdamOptimizerConfig = AdamOptimizerConfig(),
    val optimizerState: AdamOptimizerState = AdamOptimizerState(
        firstMomentEstimates = List(trainedMicrogpt.model.values().size) { 0.0 },
        secondMomentEstimates = List(trainedMicrogpt.model.values().size) { 0.0 }
    ),
    val completedStepCount: Int = 0,
    val latestLoss: Double? = null,
    val latestValidationLoss: Double? = null,
    val progressHistory: List<MicrogptTrainingProgress> = emptyList()
) {
    val isComplete: Boolean
        get() = completedStepCount >= trainingStepCount
}

data class MicrogptTrainingStepResult(
    val session: MicrogptTrainingSession,
    val progress: MicrogptTrainingProgress,
    val progressHistory: List<MicrogptTrainingProgress>
)

data class AdamUpdateResult(
    val model: TransformerModelParameters,
    val optimizerState: AdamOptimizerState
)

private data class ParameterUpdate(
    val value: Value,
    val firstMomentEstimate: Double,
    val secondMomentEstimate: Double
)

private fun Int.leftPad(width: Int): String =
    toString().padStart(width, ' ')

/**
 * Gaussian parameter initialization via Box-Muller.
 *
 * The model parameters start random and become meaningful only through training.
 * The blog describes parameters as "the knowledge of the model":
 * after optimization, the statistical structure of the training set is distilled
 * into these numbers.
 *
 * PARAMETERS:
 * 
 * mean: The center point of the distribution (default 0.0)
 * - All generated values will cluster around this value
 * - In neural networks, mean is typically 0 to avoid systematic bias
 * 
 * standardDeviation: controls the "spread" or "width" of the distribution
 * - Larger standardDeviation → values spread out more widely from the mean
 * - Smaller standardDeviation → values cluster more tightly around the mean
 * - standardDeviation = 1.0 means ~68% of values fall within [mean-1, mean+1]
 * - standardDeviation = 2.0 means values are twice as spread out
 * 
 * WHY STANDARD DEVIATION MATTERS IN NEURAL NETWORKS:
 * - Too large standardDeviation → initial weights too large → exploding activations/gradients
 * - Too small standardDeviation → initial weights too small → vanishing gradients, slow learning
 * - The 0.08 value used in matrix() is empirically tuned for this tiny model
 * - Production models often use more sophisticated initialization schemes like
 *   Xavier/Glorot or He initialization, which scale standardDeviation based on layer dimensions
 *   to maintain stable activation magnitudes through the network depth
 * 
 * Example: randomGaussian(mean=0.0, standardDeviation=0.08) generates values mostly in [-0.24, 0.24]
 */

fun randomGaussian(
    randomNumberGenerator: Random,
    mean: Double = 0.0,
    standardDeviation: Double = 1.0
): Double {
    var firstUniformSample = 0.0
    while (firstUniformSample == 0.0) {
        firstUniformSample = randomNumberGenerator.nextDouble()
    }
    val secondUniformSample = randomNumberGenerator.nextDouble()
    val standardNormalSample =
        sqrt(-2.0 * ln(firstUniformSample)) * cos(2.0 * PI * secondUniformSample)
    return mean + standardDeviation * standardNormalSample
}

/**
 * Fisher-Yates shuffle.
 *
 * The dataset order is randomized so the optimizer sees examples in mixed order.
 * This reduces harmful ordering effects during training.
 */
fun <T> List<T>.shuffledBy(randomNumberGenerator: Random): List<T> {
    val shuffled = toMutableList()
    for (currentIndex in shuffled.lastIndex downTo 1) {
        val swapIndex = randomNumberGenerator.nextInt(currentIndex + 1)
        val temporaryValue = shuffled[currentIndex]
        shuffled[currentIndex] = shuffled[swapIndex]
        shuffled[swapIndex] = temporaryValue
    }
    return shuffled
}

/**
 * Create a matrix of trainable scalar parameters.
 *
 * In this file, every learned object is ultimately just a collection of numbers.
 * The names of the matrices tell us their role.
*/
// Creates a 2D matrix of learnable parameters.
// outputSize = number of output units
// inputSize  = number of input units
//
// In deep learning, a "weight matrix" transforms one vector space into another.
// Each element is a trainable scalar parameter.
fun matrix(
    outputSize: Int,
    inputSize: Int,
    randomNumberGenerator: Random,
    standardDeviation: Double = 0.08
): Matrix =
    List(outputSize) {
        List(inputSize) { Value(randomGaussian(randomNumberGenerator, 0.0, standardDeviation)) }
    }

/**
 * Matrix-vector multiply without bias.
 *
 * This is the simplest form of a neural network layer:
 * y = W x
 *
 * The blog frames this as the "fundamental building block" of neural networks:
 * a learned linear transformation.
 *
 * VISUAL EXPLANATION:
 *
 * Input x: A vector representing the current "state" or "activations"
 * - Shape: [inputSize] (a 1D list of inputSize elements)
 * - Example: if inputSize=3, inputVector might be [0.5, -0.2, 0.8]
 * - In the context of this GPT:
 *   - Could be a token embedding: the dense vector representation of a character
 *   - Could be hidden activations from a previous layer
 *   - Each element is a learned feature or representation component
 *
 * Weight matrix: The learned transformation
 * - Shape: [outputSize × inputSize] (a 2D list with outputSize rows, each containing inputSize elements)
 * - Example: if outputSize=2 and inputSize=3, weights might be:
 *     [[0.1, 0.3, -0.2],   <- row 0: weights for computing output[0]
 *      [0.4, -0.1, 0.5]]   <- row 1: weights for computing output[1]
 * - Each row represents the weights for producing one output neuron
 * - These weights are learned during training via backpropagation
 * - They encode "what pattern in x should activate this output neuron?"
 *
 * Return value: The transformed vector
 * - Shape: [outputSize] (a 1D list of outputSize elements)
 * - Example: continuing the example above:
 *     output[0] = 0.1*0.5 + 0.3*(-0.2) + (-0.2)*0.8 = 0.05 - 0.06 - 0.16 = -0.17
 *     output[1] = 0.4*0.5 + (-0.1)*(-0.2) + 0.5*0.8 = 0.20 + 0.02 + 0.40 = 0.62
 *     result = [-0.17, 0.62]
 * - Each output element is a weighted combination of all input elements
 * - In the GPT context:
 *   - When projecting to queries/keys/values: transforms token representation
 *     into attention subspace
 *   - When projecting back from attention: combines attention head outputs
 *   - In feed-forward layers: transforms representation through nonlinear computation
 *
 * MATRIX-VECTOR MULTIPLICATION MECHANICS:
 * Given:
 *   inputVector = [x₀, x₁, ..., x_{inputSize-1}]
 *   weights = [[w₀₀, w₀₁, ..., w₀_{inputSize-1}],
 *        [w₁₀, w₁₁, ..., w₁_{inputSize-1}],
 *        ...
 *        [w_{outputSize-1,0}, w_{outputSize-1,1}, ..., w_{outputSize-1,inputSize-1}]]
 *
 * Computation:
 *   y[0] = weights[0][0]*inputVector[0] + ... + weights[0][inputSize-1]*inputVector[inputSize-1]
 *   y[1] = weights[1][0]*inputVector[0] + ... + weights[1][inputSize-1]*inputVector[inputSize-1]
 *   ...
 *   y[outputSize-1] = weights[outputSize-1][0]*inputVector[0] + ...
 *
 * CONCRETE GPT EXAMPLE:
 * When computing queries in attention:
 *   inputVector = current token's embedding vector [embeddingSize=16 elements]
 *   weights = query weight matrix model.layers[0].attention.queryWeights [16 rows × 16 columns]
 *   result = query vector [16 elements]
 *
 * The query vector encodes "what information is this position looking for?"
 * by linearly transforming the token representation through learned weights.
 *
 * WHY NO BIAS?
 * Traditional neural network layers include a bias term: y = Wx + b
 * This implementation omits bias for simplicity. Many modern architectures
 * (especially with normalization layers like RMSNorm) work well without
 * explicit bias terms in every linear layer.
 */
// Linear layer (without bias):
// y = W x
//
// Each output neuron is a dot product between one row of W and the input x.
// This is the basic building block of neural networks.

fun linear(inputVector: List<Value>, weights: List<List<Value>>): List<Value> =
    weights.map { row ->
        var outputValue = Value(0.0)
        for (columnIndex in row.indices) {
            outputValue = outputValue + row[columnIndex] * inputVector[columnIndex]
        }
        outputValue
    }

/**
 * Softmax turns arbitrary logits into probabilities.
 *
 * Logits are not probabilities:
 * - they can be negative
 * - they do not sum to 1
 * - only relative differences matter
 *
 * After softmax:
 * - every output is positive
 * - outputs sum to 1
 * - they can be interpreted as a categorical distribution over the vocabulary
 *
 * Numerical stability trick:
 * subtract max(logit) before exponentiating.
 * This does not change the probabilities mathematically.
 */
// Softmax converts raw scores ("logits") into a probability distribution.
// Logits can be any real numbers. Softmax makes them:
// - all positive
// - sum to 1
//
// This is used for next-token prediction: the model outputs one score per token,
// then softmax turns those scores into predicted probabilities.
fun softmax(logits: List<Value>): List<Value> {
    val maxLogitValue = logits.maxOf { it.data }
    val exponentials = logits.map { (it - maxLogitValue).exp() }

    var total = Value(0.0)
    for (exponential in exponentials) total += exponential

    return exponentials.map { it / total }
}

/**
 * RMSNorm = Root Mean Square Normalization.
 *
 * The blog highlights that this code uses RMSNorm instead of GPT-2's LayerNorm
 * as a simplifying substitution. It serves the same broad purpose here:
 * keep activations in a stable numeric range.
 *
 * This helps optimization because extremely large or tiny activations can make
 * gradients unstable or ineffective.
 */
// RMSNorm = Root Mean Square Normalization.
// Normalization rescales activations so they stay in a healthy numeric range.
// This often improves optimization stability.
//
// Here we compute:
// ms = mean(x^2)
// scale = 1 / sqrt(ms + eps)
// output = x * scale
//
// Unlike LayerNorm, this version only rescales and does not center.
fun rmsnorm(inputVector: List<Value>): List<Value> {
    var meanSquare = Value(0.0)
    for (value in inputVector) {
        meanSquare = meanSquare + value * value
    }
    meanSquare /= inputVector.size.toDouble()
    val scale = (meanSquare + 1e-5).pow(-0.5)
    return inputVector.map { it * scale }
}

/**
 * Sample one index from a weighted categorical distribution.
 *
 * During inference we do not always take the argmax token.
 * Instead, we often sample according to predicted probabilities so generation
 * remains varied rather than deterministic.
 */
// Sample an index according to a probability distribution.
// This is used during inference (generation), where the model predicts a
// probability for each token and we randomly choose according to those weights.
fun weightedChoice(weights: List<Double>, randomNumberGenerator: Random): Int {
    val total = weights.sum()
    var randomThreshold = randomNumberGenerator.nextDouble() * total
    for (weightIndex in weights.indices) {
        randomThreshold -= weights[weightIndex]
        if (randomThreshold <= 0.0) return weightIndex
    }
    return weights.lastIndex
}

fun createKeyValueCache(layerCount: Int): KeyValueCache =
    List(layerCount) { emptyList() }

fun runTransformerModel(
    model: TransformerModelParameters,
    config: TransformerConfig,
    tokenId: Int,
    positionId: Int,
    keys: KeyValueCache,
    values: KeyValueCache
): TransformerRun {
    val tokenEmbedding = model.tokenEmbedding[tokenId]
    val positionEmbedding = model.positionEmbedding[positionId]
    var hiddenState = tokenEmbedding.zip(positionEmbedding).map { (tokenValue, positionValue) ->
        tokenValue + positionValue
    }

    hiddenState = rmsnorm(hiddenState)
    var currentKeys = keys
    var currentValues = values

    for (layerIndex in 0 until config.layerCount) {
        val layerRun = runTransformerLayer(
            hiddenState = hiddenState,
            layer = model.layers[layerIndex],
            layerIndex = layerIndex,
            config = config,
            keys = currentKeys,
            values = currentValues
        )
        hiddenState = layerRun.hiddenState
        currentKeys = layerRun.keys
        currentValues = layerRun.values
    }

    return TransformerRun(
        logits = linear(hiddenState, model.languageModelHead),
        keys = currentKeys,
        values = currentValues
    )
}

fun runTransformerLayer(
    hiddenState: List<Value>,
    layer: TransformerLayerParameters,
    layerIndex: Int,
    config: TransformerConfig,
    keys: KeyValueCache,
    values: KeyValueCache
): TransformerLayerRun {
    var residualState = hiddenState
    var normalizedState = rmsnorm(hiddenState)

    val query = linear(normalizedState, layer.attention.queryWeights)
    val key = linear(normalizedState, layer.attention.keyWeights)
    val value = linear(normalizedState, layer.attention.valueWeights)

    val updatedKeys = keys.replacedAt(layerIndex, keys[layerIndex] + listOf(key))
    val updatedValues = values.replacedAt(layerIndex, values[layerIndex] + listOf(value))

    val attentionOutput = runMultiHeadAttention(
        query = query,
        keys = updatedKeys[layerIndex],
        values = updatedValues[layerIndex],
        config = config
    )
    var blockOutput = linear(attentionOutput, layer.attention.outputProjectionWeights)
    var updatedHiddenState = blockOutput.zip(residualState).map { (attentionValue, residualValue) ->
        attentionValue + residualValue
    }

    residualState = updatedHiddenState
    normalizedState = rmsnorm(updatedHiddenState)
    blockOutput = linear(normalizedState, layer.feedForward.expansionWeights)
    blockOutput = blockOutput.map { it.relu() }
    blockOutput = linear(blockOutput, layer.feedForward.projectionWeights)

    updatedHiddenState = blockOutput.zip(residualState).map { (feedForwardValue, residualValue) ->
        feedForwardValue + residualValue
    }
    return TransformerLayerRun(
        hiddenState = updatedHiddenState,
        keys = updatedKeys,
        values = updatedValues
    )
}

fun runMultiHeadAttention(
    query: List<Value>,
    keys: List<List<Value>>,
    values: List<List<Value>>,
    config: TransformerConfig
): List<Value> =
    (0 until config.attentionHeadCount).flatMap { headIndex ->
        val headStartIndex = headIndex * config.attentionHeadSize
        val headQuery = query.subList(headStartIndex, headStartIndex + config.attentionHeadSize)
        val headKeys = keys.map { it.subList(headStartIndex, headStartIndex + config.attentionHeadSize) }
        val headValues = values.map { it.subList(headStartIndex, headStartIndex + config.attentionHeadSize) }
        val attentionWeights = softmax(attentionLogits(headQuery, headKeys, config.attentionHeadSize))

        (0 until config.attentionHeadSize).map { headValueIndex ->
            weightedHeadValueSum(attentionWeights, headValues, headValueIndex)
        }
    }

fun attentionLogits(
    headQuery: List<Value>,
    headKeys: List<List<Value>>,
    attentionHeadSize: Int
): List<Value> =
    headKeys.map { previousKey ->
        var dotProduct = Value(0.0)
        for (headValueIndex in 0 until attentionHeadSize) {
            dotProduct = dotProduct + headQuery[headValueIndex] * previousKey[headValueIndex]
        }
        dotProduct / sqrt(attentionHeadSize.toDouble())
    }

fun weightedHeadValueSum(
    attentionWeights: List<Value>,
    headValues: List<List<Value>>,
    headValueIndex: Int
): Value {
    var weightedValueSum = Value(0.0)
    for (timeIndex in headValues.indices) {
        val weightedHeadValue = attentionWeights[timeIndex] * headValues[timeIndex][headValueIndex]
        weightedValueSum = weightedValueSum + weightedHeadValue
    }
    return weightedValueSum
}

fun trainMicrogptStep(session: MicrogptTrainingSession): MicrogptTrainingStepResult? {
    if (session.isComplete) return null

    val step = session.completedStepCount
    val document = session.documents[step % session.documents.size]
    val loss = trainOnDocument(
        model = session.trainedMicrogpt.model,
        config = session.trainedMicrogpt.config,
        tokenizer = session.trainedMicrogpt.tokenizer,
        document = document
    )

    val gradients = loss.backward()

    val update = applyAdamUpdate(
        model = session.trainedMicrogpt.model,
        gradients = gradients,
        optimizerState = session.optimizerState,
        optimizerConfig = session.optimizerConfig,
        step = step,
        trainingStepCount = session.trainingStepCount
    )
    val updatedMicrogpt = session.trainedMicrogpt.copy(model = update.model)

    val progress = MicrogptTrainingProgress(
        completedStepCount = session.completedStepCount + 1,
        trainingStepCount = session.trainingStepCount,
        loss = loss.data,
        validationLoss = null
    )
    val progressHistory = session.progressHistory + progress

    return MicrogptTrainingStepResult(
        session = session.copy(
            trainedMicrogpt = updatedMicrogpt,
            optimizerState = update.optimizerState,
            completedStepCount = progress.completedStepCount,
            latestLoss = progress.loss,
            progressHistory = progressHistory
        ),
        progress = progress,
        progressHistory = progressHistory
    )
}

fun trainOnDocument(
    model: TransformerModelParameters,
    config: TransformerConfig,
    tokenizer: CharacterTokenizer,
    document: String
): Value {
    val tokens = tokenizer.encodeDocument(document)
    val predictionStepCount = minOf(config.contextWindowSize, tokens.size - 1)
    var keys = createKeyValueCache(config.layerCount)
    var values = createKeyValueCache(config.layerCount)
    var loss = Value(0.0)

    for (positionId in 0 until predictionStepCount) {
        val tokenId = tokens[positionId]
        val targetTokenId = tokens[positionId + 1]
        val modelRun = runTransformerModel(model, config, tokenId, positionId, keys, values)
        keys = modelRun.keys
        values = modelRun.values
        val probabilities = softmax(modelRun.logits)
        val positionLoss = -probabilities[targetTokenId].log()
        loss = loss + positionLoss
    }

    return (1.0 / predictionStepCount.toDouble()) * loss
}

fun calculateDocumentLoss(
    model: TransformerModelParameters,
    config: TransformerConfig,
    tokenizer: CharacterTokenizer,
    document: String
): Double =
    trainOnDocument(
        model = model,
        config = config,
        tokenizer = tokenizer,
        document = document
    ).data

fun applyAdamUpdate(
    model: TransformerModelParameters,
    gradients: Map<Value, Double>,
    optimizerState: AdamOptimizerState,
    optimizerConfig: AdamOptimizerConfig,
    step: Int,
    trainingStepCount: Int
): AdamUpdateResult {
    val parameters = model.values()
    val stepLearningRate = optimizerConfig.learningRate * (1.0 - step.toDouble() / trainingStepCount.toDouble())

    val parameterUpdates = parameters.mapIndexed { parameterIndex, parameter ->
        val gradient = gradients[parameter] ?: 0.0
        val firstMomentEstimate =
            optimizerConfig.firstMomentDecay * optimizerState.firstMomentEstimates[parameterIndex] +
                (1.0 - optimizerConfig.firstMomentDecay) * gradient
        val secondMomentEstimate =
            optimizerConfig.secondMomentDecay * optimizerState.secondMomentEstimates[parameterIndex] +
                (1.0 - optimizerConfig.secondMomentDecay) * gradient.pow(2.0)

        val biasCorrectedFirstMoment =
            firstMomentEstimate /
                (1.0 - optimizerConfig.firstMomentDecay.pow(step + 1.0))
        val biasCorrectedSecondMoment =
            secondMomentEstimate /
                (1.0 - optimizerConfig.secondMomentDecay.pow(step + 1.0))
        val parameterUpdate =
            stepLearningRate * biasCorrectedFirstMoment /
                (sqrt(biasCorrectedSecondMoment) + optimizerConfig.epsilon)

        ParameterUpdate(
            value = Value(parameter.data - parameterUpdate),
            firstMomentEstimate = firstMomentEstimate,
            secondMomentEstimate = secondMomentEstimate
        )
    }

    return AdamUpdateResult(
        model = model.withValues(parameterUpdates.map { it.value }),
        optimizerState = AdamOptimizerState(
            firstMomentEstimates = parameterUpdates.map { it.firstMomentEstimate },
            secondMomentEstimates = parameterUpdates.map { it.secondMomentEstimate }
        )
    )
}

fun generateSamples(
    model: TransformerModelParameters,
    config: TransformerConfig,
    tokenizer: CharacterTokenizer,
    prefix: String,
    sampleCount: Int,
    temperature: Double,
    randomNumberGenerator: Random
): List<String> {
    return List(sampleCount) {
        val sample = generateSample(model, config, tokenizer, prefix, temperature, randomNumberGenerator)
        sample
    }
}

fun generateSample(
    model: TransformerModelParameters,
    config: TransformerConfig,
    tokenizer: CharacterTokenizer,
    prefix: String,
    temperature: Double,
    randomNumberGenerator: Random
): String {
    var keys = createKeyValueCache(config.layerCount)
    var values = createKeyValueCache(config.layerCount)
    var tokenId = tokenizer.sequenceBoundaryTokenId
    val normalizedPrefix = prefix
        .trim()
        .lowercase()
        .filter { character -> character in tokenizer.characterToTokenId }
        .take(config.contextWindowSize - 1)
    val sample = StringBuilder(normalizedPrefix)

    for (positionId in 0 until config.contextWindowSize) {
        val modelRun = runTransformerModel(model, config, tokenId, positionId, keys, values)
        keys = modelRun.keys
        values = modelRun.values
        if (positionId < normalizedPrefix.length) {
            tokenId = tokenizer.characterToTokenId.getValue(normalizedPrefix[positionId])
            continue
        }

        val scaledLogits = modelRun.logits.map { it / temperature }
        val probabilities = softmax(scaledLogits)

        tokenId = weightedChoice(probabilities.map { it.data }, randomNumberGenerator)
        if (tokenId == tokenizer.sequenceBoundaryTokenId) break

        sample.append(tokenizer.uniqueCharacters[tokenId])
    }

    return sample.toString()
}

fun generateSamples(
    trainedMicrogpt: TrainedMicrogpt,
    prefix: String,
    sampleCount: Int,
    temperature: Double,
    randomNumberGenerator: Random
): List<String> =
    generateSamples(
        model = trainedMicrogpt.model,
        config = trainedMicrogpt.config,
        tokenizer = trainedMicrogpt.tokenizer,
        prefix = prefix,
        sampleCount = sampleCount,
        temperature = temperature,
        randomNumberGenerator = randomNumberGenerator
    )

fun createMicrogptTrainingSession(
    inputText: String,
    randomNumberGenerator: Random,
    trainingStepCount: Int = 1000,
    validationDivisor: Int = 20
): MicrogptTrainingSession {
    /**
     * DATASET
     *
     * The blog calls data the "fuel" of language models.
     * Here the dataset is tiny and intentionally simple: one name per line.
     *
     * In production:
     * - documents are often web pages, books, code files, conversations, etc.
     * - there may be trillions of tokens
     * - substantial engineering goes into filtering, deduplication, mixing,
     *   and storage
     *
     * But algorithmically, this file already has the essential form:
     * a collection of text documents the model will learn to statistically imitate.
     */
    // Dataset:
    // each line is a training example (here, a name).
    //
    // In machine learning, the dataset is the source of examples from which
    // the model learns statistical patterns.
    val shuffledDocuments = inputText.lines()
        .map { it.trim() }
        .filter { it.isNotEmpty() }
        .shuffledBy(randomNumberGenerator)
    val validationDocumentCount = shuffledDocuments.size / validationDivisor
    val validationDocuments = shuffledDocuments.take(validationDocumentCount)
    val documents = shuffledDocuments.drop(validationDocumentCount)

    println("num training docs: ${documents.size}")
    println("num validation docs: ${validationDocuments.size}")

    /**
     * TOKENIZER
     *
     * Neural networks operate on numbers, not raw text.
     * So we map discrete symbols to integer token IDs.
     *
     * This version uses the simplest possible tokenizer:
     * one token per unique character.
     *
     * The blog makes two useful conceptual points:
     * 1) token IDs themselves carry no meaning; they are just labels
     * 2) production systems usually use subword tokenizers (e.g. BPE) for
     *    efficiency, but that changes efficiency more than essence
     */
    // Tokenizer:
    // converts characters into integer token IDs.
    //
    // Tokens are the discrete symbols the model operates on.
    // Here, each character is one token.
    val uniqueCharacters = shuffledDocuments.joinToString("").toSet().toList().sorted()

    /**
     * Sequence boundary token.
     *
     * In this tiny model, the same token is used as both:
     * - the "start a new document" marker
     * - the "end of document" marker
     *
     * So a name like "emma" becomes:
     * [boundary, e, m, m, a, boundary]
     *
     * This teaches the model both how to begin and how to stop generation.
     */
    val sequenceBoundaryTokenId = uniqueCharacters.size
    val tokenizer = CharacterTokenizer(
        uniqueCharacters = uniqueCharacters,
        sequenceBoundaryTokenId = sequenceBoundaryTokenId
    )
    // Vocabulary size = number of unique tokens the model can emit.
    val vocabularySize = tokenizer.vocabularySize
    println("vocab size: $vocabularySize")

    /**
     * MODEL HYPERPARAMETERS
     *
     * These define the size/shape of the network.
     * They are chosen by the practitioner, not learned by gradient descent.
     *
     * The blog contrasts this tiny model (~4K parameters) with real LLMs,
     * which may be hundreds of billions of parameters, but the broad layout
     * remains recognizably similar.
     */
    // layerCount         = number of Transformer blocks stacked in depth
    // embeddingSize      = embedding dimension; width of hidden vectors
    // contextWindowSize  = max context length; how many previous positions can be attended to
    // attentionHeadCount = number of attention heads
    // attentionHeadSize  = size of each head's subspace
    //
    // Hyperparameters define the architecture and training behavior but are
    // not themselves learned from data.
    val layerCount = 3
    val embeddingSize = 32
    val contextWindowSize = 16
    val attentionHeadCount = 4
    val config = TransformerConfig(
        layerCount = layerCount,
        embeddingSize = embeddingSize,
        contextWindowSize = contextWindowSize,
        attentionHeadCount = attentionHeadCount
    )

    /**
     * PARAMETER STORAGE
     *
     * The model parameters live in structured classes whose fields match the
     * architecture.
     *
     * Major pieces:
     * - wte: token embedding table
     * - wpe: positional embedding table
     * - attention projection matrices
     * - feed-forward projection matrices
     * - languageModelHead: final projection to vocabulary logits
     *
     * The blog's framing is useful here:
     * these parameters begin random, and training gradually shapes them into
     * the model's compressed "knowledge" of the dataset's statistical patterns.
     */
    // An embedding maps a discrete token ID to a dense learned vector.
    // Dense vectors allow the network to represent similarity and structure.
    val model = TransformerModelParameters.initialize(
        vocabularySize = vocabularySize,
        contextWindowSize = config.contextWindowSize,
        embeddingSize = config.embeddingSize,
        layerCount = config.layerCount,
        randomNumberGenerator = randomNumberGenerator
    )

    /**
     * ATTENTION PROJECTIONS
     *
     * Q = query, K = key, V = value
     *
     * The blog gives the useful intuition:
     * - query: "what am I looking for?"
     * - key:   "what do I contain?"
     * - value: "what information do I pass along if selected?"
     *
     * Attention is the communication mechanism:
     * it is the exact place where the current position can look back at
     * previous positions and gather information from them.
     */
    // Attention projections:
    // queryWeights = query weights
    // keyWeights = key weights
    // valueWeights = value weights
    // outputProjectionWeights = output projection
    //
    // In self-attention:
    // - queries ask "what am I looking for?"
    // - keys say "what information do I contain?"
    // - values carry the actual content to aggregate

    /**
     * FEED-FORWARD PROJECTIONS
     *
     * The feed-forward block expands the hidden representation, applies a nonlinearity,
     * then compresses it back.
     *
     * The blog describes the Transformer as alternating:
     * - communication (Attention)
     * - computation (feed-forward network)
     *
     * Attention mixes information across positions.
     * The feed-forward network then processes the current position locally.
     */
    // The position-wise feed-forward network expands hidden size to 4*embeddingSize,
    // applies nonlinearity, then projects back.
    //
    // This gives the model extra nonlinear processing capacity beyond attention.

    // Flatten all parameters into one list so the optimizer can update them.
    val parameters = model.values()
    println("num parameters: ${parameters.size}")

    return MicrogptTrainingSession(
        trainedMicrogpt = TrainedMicrogpt(
            model = model,
            config = config,
            tokenizer = tokenizer
        ),
        documents = documents,
        validationDocuments = validationDocuments,
        trainingStepCount = trainingStepCount
    )
}
