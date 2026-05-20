package org.jshmrsn.microgpt.app

import microgpt_kotlin_visualized.shared.generated.resources.Res
import org.jetbrains.compose.resources.ExperimentalResourceApi
import org.jshmrsn.microgpt.lib.microgpt
import kotlin.random.Random

@OptIn(ExperimentalResourceApi::class)
suspend fun loadMicrogptInputText(): String =
    Res.readBytes("files/input.txt").decodeToString()

suspend fun runMicrogptDemo(randomNumberGenerator: Random = Random(0)) {
    val inputText = loadMicrogptInputText()
    microgpt(inputText, randomNumberGenerator)
}
