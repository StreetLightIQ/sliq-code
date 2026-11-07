package com.streetlightiq.edge.camera

import android.content.Context
import android.graphics.*
import android.hardware.camera2.*
import android.os.SystemClock
import android.util.Size
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.streetlightiq.edge.domain.*
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.*
import java.io.File
import java.util.concurrent.Executors

data class CameraFrame(val bitmap:Bitmap,val sensorNs:Long,val receiptNs:Long,val rotation:Int,
    val clockRealtime:Boolean,val cameraContext:JsonObject)
@androidx.annotation.OptIn(androidx.camera.camera2.interop.ExperimentalCamera2Interop::class)
class CameraController(private val context:Context) {
    private val executor=Executors.newSingleThreadExecutor()
    private var provider:ProcessCameraProvider?=null
    private var video:VideoCapture<Recorder>?=null
    private var recording:Recording?=null
    private val metadata=linkedMapOf<Long,JsonObject>()
    private var cameraId:String?=null
    private var clockRealtime=false
    private var cameraStatic=JsonObject(emptyMap())
    var onFrame:suspend (CameraFrame)->Unit = {}
    var onError:(String)->Unit = {}
    var onReady:(Boolean)->Unit = {}
    var onVideoFinalized:(String,Boolean)->Unit = { _,_-> }
    private var bindGeneration=0
    @Volatile var onCapture:(JsonObject)->Unit = {}
    private var lastCaptureReceiptNs=0L
    private var previewUse:Preview?=null
    private var analysisUse:ImageAnalysis?=null
    private val mainHandler=android.os.Handler(android.os.Looper.getMainLooper())
    private fun errorName(code:Int)=when(code) {
        CameraState.ERROR_MAX_CAMERAS_IN_USE -> "too many cameras open"; CameraState.ERROR_CAMERA_IN_USE -> "camera in use"
        CameraState.ERROR_OTHER_RECOVERABLE_ERROR -> "recoverable error"; CameraState.ERROR_STREAM_CONFIG -> "settings not supported"
        CameraState.ERROR_CAMERA_DISABLED -> "camera disabled"; CameraState.ERROR_CAMERA_FATAL_ERROR -> "camera failed"
        CameraState.ERROR_DO_NOT_DISTURB_MODE_ENABLED -> "do-not-disturb"; else -> "unknown" }
    private fun wide()=ResolutionSelector.Builder()
        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
        .setResolutionStrategy(ResolutionStrategy(Size(1280,720),ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build()

    /** Keep frames upright for the current screen rotation (Surface.ROTATION_*). The survey locks the screen, so this is fixed while collecting. */
    fun setRotation(rotation:Int) {
        previewUse?.targetRotation=rotation
        analysisUse?.targetRotation=rotation
        video?.targetRotation=rotation
    }

    @androidx.annotation.OptIn(androidx.camera.camera2.interop.ExperimentalCamera2Interop::class)
    fun bind(owner:LifecycleOwner,view:PreviewView,enableVideo:Boolean) {
        val generation=++bindGeneration
        view.scaleType=PreviewView.ScaleType.FILL_CENTER  // landscape box is exactly 16:9, so this only crops the portrait setup preview
        view.implementationMode=PreviewView.ImplementationMode.COMPATIBLE
        val future=ProcessCameraProvider.getInstance(context)
        future.addListener({
            if(generation!=bindGeneration) return@addListener
            try {
                val p=future.get(); provider=p; p.unbindAll()
                val rotation=view.display?.rotation ?: android.view.Surface.ROTATION_90
                val previewBuilder=Preview.Builder().setResolutionSelector(wide()).setTargetRotation(rotation)
                val preview=previewBuilder.build().also { it.setSurfaceProvider(view.surfaceProvider) }
                previewUse=preview
                val builder=ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setResolutionSelector(wide()).setTargetRotation(rotation)
                Camera2Interop.Extender(builder).setSessionCaptureCallback(object:CameraCaptureSession.CaptureCallback() {
                    override fun onCaptureCompleted(session:CameraCaptureSession,request:CaptureRequest,result:TotalCaptureResult) {
                        val ts=result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
                        val gains=result.get(CaptureResult.COLOR_CORRECTION_GAINS)
                        val data=obj("sensor_timestamp_ns" to j(ts),"capture_frame_number" to j(result.frameNumber),
                            "exposure_time_ns" to j(result.get(CaptureResult.SENSOR_EXPOSURE_TIME)),
                            "iso" to j(result.get(CaptureResult.SENSOR_SENSITIVITY)),
                            "focal_length_mm" to j(result.get(CaptureResult.LENS_FOCAL_LENGTH)),
                            "aperture_f_number" to j(result.get(CaptureResult.LENS_APERTURE)),
                            "awb_state" to j(result.get(CaptureResult.CONTROL_AWB_STATE)),
                            "ae_state" to j(result.get(CaptureResult.CONTROL_AE_STATE)),
                            "white_balance_gains" to (gains?.let { JsonArray(listOf(it.red,it.greenEven,it.greenOdd,it.blue).map(::j)) } ?: JsonNull))
                        val all=JsonObject(data+mapOf("all_result_keys" to CameraMetadata.capture(result)))
                        val now=SystemClock.elapsedRealtimeNanos()
                        if(now-lastCaptureReceiptNs>=100000000L) { lastCaptureReceiptNs=now; onCapture(all) }
                        synchronized(metadata) { metadata[ts]=all; while(metadata.size>120) metadata.remove(metadata.keys.first()) }
                    }
                })
                val analysis=builder.build()
                analysisUse=analysis
                analysis.setAnalyzer(executor) { image ->
                    val receipt=SystemClock.elapsedRealtimeNanos()
                    try {
                        val data=synchronized(metadata) { metadata[image.imageInfo.timestamp] }
                        val bitmap=convert(image)
                        try {
                            runBlocking { onFrame(CameraFrame(bitmap,image.imageInfo.timestamp,receipt,image.imageInfo.rotationDegrees,
                                clockRealtime,JsonObject(cameraStatic+mapOf("capture" to (data ?: JsonNull),"metadata_exact_frame_match" to j(data!=null),
                                    "analysis_buffer_width" to j(image.width),"analysis_buffer_height" to j(image.height),
                                    "analysis_crop_rect" to CameraMetadata.value(image.cropRect),"lens_facing" to j("rear"),"format" to j("RGBA_8888"))))) }
                        } finally { bitmap.recycle() }
                    } catch(e:Exception) { onError(e.message ?: "Camera frame failed") }
                    finally { image.close() }
                }
                fun group(withVideo:Boolean):UseCaseGroup {
                    val group=UseCaseGroup.Builder().addUseCase(preview).addUseCase(analysis)
                    if(withVideo) {
                        val recorder=Recorder.Builder().setQualitySelector(QualitySelector.from(Quality.SD,FallbackStrategy.lowerQualityOrHigherThan(Quality.SD))).build()
                        video=VideoCapture.Builder(recorder).setTargetRotation(rotation).build(); group.addUseCase(video!!)
                    } else video=null
                    return group.build()
                }
                val camera=try { p.bindToLifecycle(owner,CameraSelector.DEFAULT_BACK_CAMERA,group(enableVideo)) }
                catch(e:Exception) {
                    if(!enableVideo) throw e
                    p.unbindAll(); onError("This camera cannot run video and analysis together. Disable local video in Settings.")
                    p.bindToLifecycle(owner,CameraSelector.DEFAULT_BACK_CAMERA,group(false))
                }
                cameraId=Camera2CameraInfo.from(camera.cameraInfo).cameraId
                val manager=context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
                val chars=manager.getCameraCharacteristics(cameraId!!)
                clockRealtime=chars.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)==CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
                val intrinsics=chars.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
                val size=chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                cameraStatic=obj("camera_id" to j(cameraId),"timestamp_source" to j(if(clockRealtime) "REALTIME" else "UNKNOWN"),
                    "sensor_physical_width_mm" to j(size?.width),"sensor_physical_height_mm" to j(size?.height),
                    "intrinsics_sensor_pixels" to (intrinsics?.let { JsonArray(it.map(::j)) } ?: JsonNull),
                    "all_characteristics" to CameraMetadata.characteristics(chars),
                    "intrinsics_note" to j("Sensor coordinates; not calibrated to the cropped analysis stream."))
                // errors during a rebind are often transient, so act only if still closed after 3 s
                camera.cameraInfo.cameraState.removeObservers(owner)
                camera.cameraInfo.cameraState.observe(owner) { state ->
                    val error=state.error ?: return@observe
                    mainHandler.postDelayed({
                        if(generation!=bindGeneration || camera.cameraInfo.cameraState.value?.type==CameraState.Type.OPEN) return@postDelayed
                        val what="${error.code}: ${errorName(error.code)}"
                        onError("Camera error $what; stop and reopen the camera.")
                    },3000)
                }
                onReady(true)
            } catch(e:Exception) { onReady(false); onError(e.message ?: "Could not open rear camera") }
        },ContextCompat.getMainExecutor(context))
    }
    private fun convert(image:ImageProxy):Bitmap {
        val plane=image.planes[0]; val buffer=plane.buffer.duplicate()
        val pixels=IntArray(image.width*image.height)
        for(y in 0 until image.height) for(x in 0 until image.width) {
            val offset=y*plane.rowStride+x*plane.pixelStride
            val r=buffer.get(offset).toInt() and 255; val g=buffer.get(offset+1).toInt() and 255
            val b=buffer.get(offset+2).toInt() and 255; val a=buffer.get(offset+3).toInt() and 255
            pixels[y*image.width+x]=Color.argb(a,r,g,b)
        }
        val raw=Bitmap.createBitmap(pixels,image.width,image.height,Bitmap.Config.ARGB_8888)
        val crop=image.cropRect
        val cropped=Bitmap.createBitmap(raw,crop.left,crop.top,crop.width(),crop.height())
        val rotated=if(image.imageInfo.rotationDegrees==0) cropped else Bitmap.createBitmap(cropped,0,0,cropped.width,cropped.height,
            Matrix().apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) },true)
        if(cropped !== raw && cropped !== rotated) cropped.recycle()
        if(raw !== rotated) raw.recycle()
        return rotated
    }
    fun startVideo(file:File) {
        val capture=video ?: error("Local video unavailable; disable it in Settings.")
        file.parentFile?.mkdirs()
        recording=capture.output.prepareRecording(context,FileOutputOptions.Builder(file).build())
            .start(ContextCompat.getMainExecutor(context)) { event ->
                if(event is VideoRecordEvent.Finalize) { recording=null; onVideoFinalized(file.absolutePath,!event.hasError()) }
            }
    }
    fun stopVideo() { recording?.stop(); recording=null }
    fun unbind() { bindGeneration++; stopVideo(); provider?.unbindAll(); onReady(false) }
}
