package com.streetlightiq.edge.domain

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.*
import kotlin.math.*

val wireJson = Json { encodeDefaults = true; explicitNulls = true; ignoreUnknownKeys = true }
fun obj(vararg pairs: Pair<String, JsonElement>) = JsonObject(mapOf(*pairs))
fun j(value: String?) : JsonElement = value?.let { JsonPrimitive(it) } ?: JsonNull
/** Only plain finite numbers become JSON numbers. Anything else that extends Number (android.util.Rational prints as "109/64",
 *  NaN, Infinity) would make the whole upload invalid JSON, so it is converted to its value or null. */
fun j(value: Number?) : JsonElement = when(value) {
    null -> JsonNull
    is Int, is Long, is Short, is Byte -> JsonPrimitive(value)
    is Double -> if(value.isFinite()) JsonPrimitive(value) else JsonNull
    is Float -> if(value.isFinite()) JsonPrimitive(value) else JsonNull
    else -> value.toDouble().let { if(it.isFinite()) JsonPrimitive(it) else JsonNull }
}

private val jsonNumber=Regex("-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?")
private val fraction=Regex("(-?[0-9]+)/(-?[0-9]+)")
/** Last line of defence before anything is uploaded or logged: every unquoted value must be valid JSON.
 *  "a/b" fractions become their decimal value, NaN/Infinity and any other invalid literal become null. */
fun wireSafe(e: JsonElement): JsonElement = when(e) {
    is JsonObject -> JsonObject(e.mapValues { wireSafe(it.value) })
    is JsonArray -> JsonArray(e.map { wireSafe(it) })
    is JsonNull -> e
    is JsonPrimitive -> when {
        e.isString -> e
        e.content=="true" || e.content=="false" -> e
        jsonNumber.matches(e.content) -> e
        else -> fraction.matchEntire(e.content)?.let { m ->
            val d=m.groupValues[2].toDouble(); if(d==0.0) JsonNull else (m.groupValues[1].toDouble()/d).let { if(it.isFinite()) JsonPrimitive(it) else JsonNull }
        } ?: JsonNull
    }
}
fun j(value: Boolean) : JsonElement = JsonPrimitive(value)
inline fun <reified T> encoded(value: T): JsonElement = wireJson.encodeToJsonElement(value)

@Serializable
data class Box(val x1: Float, val y1: Float, val x2: Float, val y2: Float) {
    val cx get() = (x1 + x2) / 2
    val cy get() = (y1 + y2) / 2
    val width get() = max(0f, x2 - x1)
    val height get() = max(0f, y2 - y1)
    val area get() = width * height
    fun iou(b: Box): Float {
        val intersection = max(0f, min(x2,b.x2)-max(x1,b.x1)) * max(0f,min(y2,b.y2)-max(y1,b.y1))
        return intersection / max(1e-9f, area+b.area-intersection)
    }
    fun expanded(amount: Float) = Box((x1-amount).coerceAtLeast(0f),(y1-amount).coerceAtLeast(0f),
        (x2+amount).coerceAtMost(1f),(y2+amount).coerceAtMost(1f))
    fun contains(x:Float,y:Float) = x in x1..x2 && y in y1..y2
}

@Serializable
data class Detection(val bbox: Box, val score: Float, val class_id: Int,
    val class_name: String, val source: String, val score_kind: String = "model_confidence")

@Serializable
data class FrameTime(val frame_id: Long, val utc_ms: Long, val monotonic_ns: Long,
    val sensor_timestamp_ns: Long, val timing_basis: String, val rotation_degrees: Int,
    val width: Int, val height: Int)

@Serializable
data class LocationSample(val id: String, val latitude: Double, val longitude: Double,
    val accuracy_m: Float, val utc_ms: Long, val monotonic_ns: Long,
    val altitude_m: Double?, val vertical_accuracy_m: Float?, val speed_mps: Float?,
    val speed_accuracy_mps: Float?, val bearing_degrees: Float?, val bearing_accuracy_degrees: Float?,
    val provider: String?, val is_mock: Boolean)

@Serializable
data class TrackSample(val observation_id: String, val track_id: Long, val frame: FrameTime,
    val detection: Detection, val device_location: LocationSample?, val gps_time_offset_ms: Long?,
    val gps_quality: String)

data class Track(val id: Long, val observationId: String, var detection: Detection,
    val firstTime: FrameTime, var lastTime: FrameTime, var matches: Int = 1,
    var vx: Float = 0f, var vy: Float = 0f, var misses: Int = 0,
    var scoreSum: Double = detection.score.toDouble(), var minScore: Float = detection.score,
    var maxScore: Float = detection.score, val firstBox: Box = detection.bbox,
    var maxAreaBox: Box = detection.bbox, var revision: Long = 0,
    var lastCheckpointNs: Long = 0, var lastCloudSampleNs: Long = 0,
    val history: ArrayDeque<TrackSample> = ArrayDeque())

data class TrackingResult(val active: List<Track>, val matched: List<Track>, val completed: List<Track>)
data class DetectionResult(val targets: List<Detection>, val inferenceMs: Double)
data class OverlayBox(val box: Box, val label: String)

enum class SessionState { IDLE, INITIALIZING, RUNNING, PAUSED, STOPPING, COMPLETED, ERROR }
data class PipelineUi(val state: SessionState = SessionState.IDLE, val sessionId: String? = null,
    val elapsedMs: Long = 0, val tracks: Int = 0, val activeTracks: Int = 0,
    val fps: Double = 0.0, val latencyMs: Double = 0.0, val width: Int = 3, val height: Int = 4,
    val boxes: List<OverlayBox> = emptyList(), val error: String? = null, val modelReady: Boolean = false,
    val videoPath: String? = null)
