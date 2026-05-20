package org.jshmrsn.microgpt.app

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform