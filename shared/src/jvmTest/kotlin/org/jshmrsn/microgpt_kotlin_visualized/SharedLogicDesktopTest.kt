package org.jshmrsn.microgpt.app

import org.jshmrsn.microgpt.lib.AdamOptimizerConfig
import org.jshmrsn.microgpt.lib.TransformerConfig
import org.jshmrsn.microgpt.lib.calculateDocumentLoss
import org.jshmrsn.microgpt.lib.createMicrogptTrainingSession
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SharedLogicDesktopTest {

    @Test
    fun transformerConfigRequiresHeadsToDivideEmbeddingSize() {
        assertFailsWith<IllegalArgumentException> {
            TransformerConfig(
                layerCount = 1,
                embeddingSize = 10,
                contextWindowSize = 8,
                attentionHeadCount = 3
            )
        }
    }

    @Test
    fun trainingReducesAverageLossOnToyCharacterCorpus() {
        var session = createMicrogptTrainingSession(
            inputDocuments = listOf(
                "hello there",
                "hello world",
                "good night",
                "good morning",
                "tiny model",
                "tiny world"
            ),
            randomNumberGenerator = Random(1),
            trainingStepCount = 60,
            validationSetDivisor = 3,
            validationEvaluationDocumentCount = 1,
            transformerConfig = TransformerConfig(
                layerCount = 1,
                embeddingSize = 12,
                contextWindowSize = 16,
                attentionHeadCount = 3
            ),
            optimizerConfig = AdamOptimizerConfig(
                learningRate = 0.003,
                firstMomentDecay = 0.9,
                secondMomentDecay = 0.999,
                epsilon = 1e-8
            )
        )

        fun averageTrainingLoss(): Double =
            session.documents
                .map { document ->
                    calculateDocumentLoss(
                        model = session.trainedMicrogpt.model,
                        config = session.trainedMicrogpt.config,
                        tokenizer = session.trainedMicrogpt.tokenizer,
                        document = document
                    )
                }
                .average()

        val initialLoss = averageTrainingLoss()

        repeat(session.trainingStepCount) {
            session = org.jshmrsn.microgpt.lib.trainMicrogptStep(session)?.session ?: session
        }

        val finalLoss = averageTrainingLoss()
        assertTrue(finalLoss < initialLoss, "expected final loss $finalLoss to be lower than initial loss $initialLoss")
    }
}
