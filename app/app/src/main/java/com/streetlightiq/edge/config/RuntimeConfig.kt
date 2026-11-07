package com.streetlightiq.edge.config

import kotlinx.serialization.Serializable

@Serializable
data class RuntimeConfig(
    val version: String = "sliq-1.0",
    val targetFps: Int = 5,
    val maxMissingMs: Long = 1500,
    val centreGate: Float = 0.16f,
    val maxTracks: Int = 48,
    val maxHistory: Int = 128,
    val gpsMaxAgeMs: Long = 5000,
    val gpsStartAccuracyM: Float = 50f,
    val checkpointMs: Long = 2000,
    val cloudTrackSampleMs: Long = 1000,
    val saveEvidence: Boolean = true,
    val saveContextFrames: Boolean = true,
    val saveLocalVideo: Boolean = false,
    val showOverlay: Boolean = true,
    val allowMissingGps: Boolean = false
)

@Serializable
data class ModelMetadata(val schema_version: Int = 1, val model_version: String,
    val sha256: String, val file: String = "model/student_n_int8.tflite", val input_width: Int,
    val input_height: Int, val input_color_space: String = "RGB", val normalization: String = "0_1",
    val output_layout: String = "channels_first", val output_coordinate_units: String = "normalized",
    val labels: List<String>, val threshold: Float = 0.25f, val nms_iou: Float = 0.45f,
    val quantization: String = "INT8", val num_threads: Int = 4,
    val input_layout:String="NHWC", val output_format:String="raw_xywh", val target_labels:List<String> = emptyList(),
    val provenance:Map<String,String> = emptyMap(), val family:String="yolo26",
    val mean:List<Float> = listOf(0f,0f,0f), val std:List<Float> = listOf(1f,1f,1f))
