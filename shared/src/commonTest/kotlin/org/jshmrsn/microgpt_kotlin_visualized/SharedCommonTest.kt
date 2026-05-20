package org.jshmrsn.microgpt.app

import org.jshmrsn.microgpt.lib.Value
import org.jshmrsn.microgpt.lib.createMicrogptTrainingSession
import org.jshmrsn.microgpt.lib.trainMicrogptStep
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotSame
import kotlin.test.assertNotNull

class SharedCommonTest {

    @Test
    fun example() {
        assertEquals(3, 1 + 2)
    }

    @Test
    fun backwardReturnsGradientsWithoutMutatingValues() {
        val x = Value(2.0)
        val y = Value(3.0)
        val z = x * y + x

        val gradients = z.backward()

        assertEquals(4.0, gradients.getValue(x))
        assertEquals(2.0, gradients.getValue(y))
    }

    @Test
    fun trainingStepReturnsNewSession() {
        val session = createMicrogptTrainingSession(
            inputText = "1+1=2\n2+2=4\n",
            randomNumberGenerator = kotlin.random.Random(0),
            trainingStepCount = 1
        )

        val result = assertNotNull(trainMicrogptStep(session))

        assertEquals(0, session.completedStepCount)
        assertEquals(1, result.session.completedStepCount)
        assertNotSame(session, result.session)
        assertNotSame(session.trainedMicrogpt.model, result.session.trainedMicrogpt.model)
    }
}
