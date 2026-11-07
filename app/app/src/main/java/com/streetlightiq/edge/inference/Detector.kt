package com.streetlightiq.edge.inference

import android.content.Context
import android.graphics.*
import com.streetlightiq.edge.config.*
import com.streetlightiq.edge.domain.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.*

interface Detector {
    fun initialize()
    fun detect(bitmap: Bitmap): DetectionResult
    fun close()
    val metadata: ModelMetadata
    val runtimeId: String
}

/** CPU LiteRT detector for the bundled student model (assets/model). */
class TfliteYoloDetector(private val context: Context) : Detector {
    override val runtimeId = "litert-1.4.2-cpu-xnnpack"
    override lateinit var metadata: ModelMetadata
        private set
    private var interpreter: Interpreter? = null
    private lateinit var input: ByteBuffer
    private lateinit var output: ByteBuffer
    private var channels = 0
    private var anchors = 0
    private var inputType = DataType.INT8
    private var outputType = DataType.INT8
    private var inputScale = 1f
    private var inputZero = 0
    private var outputScale = 1f
    private var outputZero = 0
    private lateinit var canvasBitmap: Bitmap
    private lateinit var pixels: IntArray

    override fun initialize() {
        if (interpreter != null) return
        metadata = wireJson.decodeFromString(context.assets.open("model/model_metadata.json").bufferedReader().use { it.readText() })
        require(metadata.mean==listOf(0f,0f,0f) && metadata.std==listOf(1f,1f,1f)) { "Detector preprocessing must be RGB / 255" }
        require(metadata.input_color_space == "RGB" && metadata.normalization == "0_1")
        require(metadata.output_coordinate_units in setOf("normalized","pixels"))
        val bytes = context.assets.open(metadata.file).use { it.readBytes() }
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        require(digest == metadata.sha256) { "Model checksum does not match model_metadata.json" }
        val model = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply { put(bytes); rewind() }
        val it = Interpreter(model, Interpreter.Options().setNumThreads(metadata.num_threads).setUseXNNPACK(true))
        try {
            val ti = it.getInputTensor(0)
            val to = it.getOutputTensor(0)
            val expected=if(metadata.input_layout=="NCHW") intArrayOf(1,3,metadata.input_height,metadata.input_width) else intArrayOf(1,metadata.input_height,metadata.input_width,3)
            require(metadata.input_layout in setOf("NHWC","NCHW") && ti.shape().contentEquals(expected)) { "Input shape does not match RGB model metadata" }
            require(metadata.labels.isNotEmpty() && metadata.input_width in 32..2048 && metadata.input_height in 32..2048)
            require(metadata.threshold in 0f..1f && metadata.num_threads in 1..8)
            require(it.inputTensorCount == 1 && it.outputTensorCount == 1 && to.shape().size == 3) { "Unsupported export; supply a detector adapter" }
            inputType = ti.dataType(); outputType = to.dataType()
            require(inputType in listOf(DataType.FLOAT32,DataType.INT16,DataType.INT8,DataType.UINT8))
            require(outputType in listOf(DataType.FLOAT32,DataType.INT16,DataType.INT8,DataType.UINT8))
            inputScale = ti.quantizationParams().scale; inputZero = ti.quantizationParams().zeroPoint
            outputScale = to.quantizationParams().scale; outputZero = to.quantizationParams().zeroPoint
            if (inputType in listOf(DataType.INT8,DataType.UINT8,DataType.INT16)) require(inputScale > 0)
            if (outputType in listOf(DataType.INT8,DataType.UINT8,DataType.INT16)) require(outputScale > 0)
            when(metadata.output_layout) {
                "channels_first" -> { channels = to.shape()[1]; anchors = to.shape()[2] }
                "channels_last" -> { channels = to.shape()[2]; anchors = to.shape()[1] }
                else -> error("Unsupported output layout")
            }
            require((metadata.output_format=="raw_xywh" && channels==4+metadata.labels.size) ||
                (metadata.output_format=="end2end_xyxy" && channels==6)) { "Unsupported YOLO output format/shape" }
            input = ByteBuffer.allocateDirect(ti.numBytes()).order(ByteOrder.nativeOrder())
            output = ByteBuffer.allocateDirect(to.numBytes()).order(ByteOrder.nativeOrder())
            canvasBitmap = Bitmap.createBitmap(metadata.input_width,metadata.input_height,Bitmap.Config.ARGB_8888)
            pixels = IntArray(metadata.input_width*metadata.input_height)
            // fail early on unsupported operators
            it.run(input,output)
            interpreter = it
        } catch(e:Exception) { it.close(); throw e }
    }

    override fun detect(bitmap: Bitmap): DetectionResult {
        val started = System.nanoTime()
        val w = metadata.input_width; val h = metadata.input_height
        val scale = min(w.toFloat()/bitmap.width,h.toFloat()/bitmap.height)
        val padX = (w-bitmap.width*scale)/2; val padY = (h-bitmap.height*scale)/2
        Canvas(canvasBitmap).apply {
            drawColor(Color.rgb(114,114,114))
            drawBitmap(bitmap,null,RectF(padX,padY,padX+bitmap.width*scale,padY+bitmap.height*scale),Paint(Paint.FILTER_BITMAP_FLAG))
        }
        canvasBitmap.getPixels(pixels,0,w,0,0,w,h)
        input.rewind()
        fun putChannel(channel:Int) {
            val real = channel / 255f
            when(inputType) {
                DataType.FLOAT32 -> input.putFloat(real)
                DataType.INT16 -> input.putShort((real/inputScale+inputZero).roundToInt().coerceIn(-32768,32767).toShort())
                DataType.INT8 -> input.put((real/inputScale+inputZero).roundToInt().coerceIn(-128,127).toByte())
                else -> input.put((real/inputScale+inputZero).roundToInt().coerceIn(0,255).toByte())
            }
        }
        if(metadata.input_layout=="NHWC") for(p in pixels) { putChannel((p shr 16) and 255); putChannel((p shr 8) and 255); putChannel(p and 255) }
        else for(c in 0..2) for(p in pixels) putChannel((p shr (16-8*c)) and 255)
        input.rewind(); output.rewind()
        checkNotNull(interpreter).run(input,output)
        fun value(c:Int,a:Int):Float {
            val index = if(metadata.output_layout == "channels_first") c*anchors+a else a*channels+c
            return when(outputType) {
                DataType.FLOAT32 -> output.getFloat(index*4)
                DataType.INT16 -> (output.getShort(index*2).toInt()-outputZero)*outputScale
                DataType.INT8 -> (output.get(index).toInt()-outputZero)*outputScale
                else -> ((output.get(index).toInt() and 255)-outputZero)*outputScale
            }
        }
        val proposed=YoloDecoder.decode(metadata,anchors,::value,bitmap.width,bitmap.height,scale,padX,padY)
        val kept=if(metadata.output_format=="end2end_xyxy") proposed else nms(proposed,metadata.nms_iou)
        val targets=kept.filter { metadata.target_labels.isEmpty() || it.class_name in metadata.target_labels }
        return DetectionResult(targets,(System.nanoTime()-started)/1e6)
    }

    private fun nms(items: List<Detection>, threshold: Float): List<Detection> {
        val kept = ArrayList<Detection>()
        for(d in items.sortedByDescending { it.score }.take(1000)) {
            if(kept.none { it.class_id == d.class_id && it.bbox.iou(d.bbox) > threshold }) kept += d
            if(kept.size >= 64) break
        }
        return kept
    }
    override fun close() { interpreter?.close(); interpreter=null; if(::canvasBitmap.isInitialized) canvasBitmap.recycle() }
}
