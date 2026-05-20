package org.jshmrsn.microgpt.app

import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState

fun main() = application {
    Window(
        onCloseRequest = ::exitApplication,
        title = "microgpt",
        state = rememberWindowState(
            size = DpSize(1600.dp, 1000.dp)
        )
    ) {
        App()
    }
}