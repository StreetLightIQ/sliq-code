package com.streetlightiq.edge.pipeline

import android.content.Context
import com.streetlightiq.edge.sensors.SurveyRecorder
import android.os.*
import android.os.Build
import com.streetlightiq.edge.BuildConfig
import com.streetlightiq.edge.camera.CameraFrame
import com.streetlightiq.edge.config.*
import com.streetlightiq.edge.data.local.*
import com.streetlightiq.edge.data.remote.TokenVault
import com.streetlightiq.edge.domain.*
import com.streetlightiq.edge.inference.*
import com.streetlightiq.edge.location.*
import com.streetlightiq.edge.evidence.*
import com.streetlightiq.edge.sync.SyncScheduler
import com.streetlightiq.edge.tracking.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.json.*
import java.io.File
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.Executors
import kotlin.math.*

/** All mutable pipeline state is confined to one executor. Camera and network are independent. */
class EdgePipeline(private val context:Context,private val preferences:Preferences,
    private val records:RecordRepository,private val locations:LocationProvider,
    private val vault:TokenVault,private val sync:SyncScheduler) {
    private val dispatcher=Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private val _ui=MutableStateFlow(PipelineUi())
    val ui:StateFlow<PipelineUi> = _ui
    private var detector:Detector=TfliteYoloDetector(context)
    private val recorder=SurveyRecorder(context)
    private var archiveFiles=emptyList<File>()
    val modelLabel=MutableStateFlow("")
    fun rawLocation(sample:LocationSample) { recorder.event("location",encoded(sample).jsonObject,sample.monotonic_ns) }
    fun capture(data:JsonObject) { recorder.event("camera_capture",data) }
    private var config=RuntimeConfig()
    private var tracker:Tracker=CentreScaleTracker(config)
    private var evidence:EvidenceSelector=BestCropSelector(true)
    private var physicsFrames=PhysicsFramePicker()
    private var savePhysicsFrames=true
    private var sessionId:String?=null
    private var ownerId=""
    private var startedUtc=0L
    private var startedNs=0L
    private var frameIndex=0L
    private var received=0L
    private var skipped=0L
    private var errors=0L
    private var consecutiveErrors=0
    private var sessionRevision=0L
    private var lastProcessedNs=0L
    private var lastBatchNs=0L
    private var lastSessionNs=0L
    private var lastLocationId:String?=null
    private var pausedAtNs:Long?=null
    private var pausedDurationNs=0L
    private var videoFile:String?=null
    private var videoFinalized:Boolean?=null
    private var lastSavedState="IDLE"
    private var lastSavedEndMs:Long?=null
    private val seenTracks=mutableSetOf<String>()
    private val cloudSamples=mutableListOf<TrackSample>()
    private val route=mutableListOf<LocationSample>()
    private val latency=ArrayDeque<Double>()
    private var lastTrackingMs=0.0
    private var lastPipelineMs=0.0

    suspend fun initialize()=withContext(dispatcher) {
        try {
            records.recover(context.filesDir)
            detector.initialize()
            modelLabel.value=detector.metadata.model_version+" · "+detector.metadata.quantization
            _ui.value=_ui.value.copy(modelReady=true,error=null)
            sync.periodic(); sync.immediate()
        } catch(e:Exception) { _ui.value=_ui.value.copy(state=SessionState.ERROR,error=e.message,modelReady=false) }
    }
    suspend fun start(cameraReady:Boolean)=withContext(dispatcher) {
        if(_ui.value.state in setOf(SessionState.RUNNING,SessionState.PAUSED,SessionState.INITIALIZING)) return@withContext
        val settings=preferences.get(); config=settings.runtime
        try {
            _ui.value=_ui.value.copy(state=SessionState.INITIALIZING,error=null)
            require(cameraReady) { "Wait for the rear camera preview." }
            require(_ui.value.modelReady) { "Model is not ready." }
            val account=vault.read() ?: error("Open Settings and sign in to your survey backend first.")
            require(account.projectUrl==settings.backend.url) { "Sign in to the configured backend." }
            require(StatFs(context.filesDir.path).availableBytes>100L*1024*1024) { "Free at least 100 MB of storage." }
            val now=SystemClock.elapsedRealtimeNanos()
            val fix=locations.buffer.closest(now,config.gpsMaxAgeMs)
            require(config.allowMissingGps || (fix.sample!=null && !fix.sample.is_mock && fix.sample.accuracy_m<=config.gpsStartAccuracyM)) {
                "Wait outdoors for a recent, real location fix within ${config.gpsStartAccuracyM.toInt()} m."
            }
            sessionId=UUID.randomUUID().toString(); ownerId=account.userId
            startedUtc=System.currentTimeMillis(); startedNs=now
            frameIndex=0; received=0; skipped=0; errors=0; consecutiveErrors=0; sessionRevision=0
            lastProcessedNs=0; lastBatchNs=now; lastSessionNs=now; lastLocationId=null
            pausedAtNs=null; pausedDurationNs=0; videoFile=null; videoFinalized=null
            seenTracks.clear(); cloudSamples.clear(); route.clear(); latency.clear()
            tracker=CentreScaleTracker(config); evidence=BestCropSelector(config.saveEvidence,config.saveContextFrames)
            physicsFrames=PhysicsFramePicker(); savePhysicsFrames=config.saveContextFrames
            archiveFiles=emptyList()
            recorder.start(sessionId!!,encoded(detector.metadata).jsonObject,encoded(config).jsonObject)
            saveSession("RUNNING",null,null)
            _ui.value=PipelineUi(state=SessionState.RUNNING,sessionId=sessionId,modelReady=true)
            sync.immediate()
        } catch(e:Exception) { recorder.finish("start_failed"); sessionId=null; _ui.value=_ui.value.copy(state=SessionState.ERROR,error=e.message) }
    }
    suspend fun pause()=withContext(dispatcher) {
        if(_ui.value.state!=SessionState.RUNNING) return@withContext
        tracker.flush().forEach { checkpoint(it,true,"paused") }
        recorder.event("survey_state",obj("state" to j("PAUSED"))); flushBatches(); pausedAtNs=SystemClock.elapsedRealtimeNanos()
        saveSession("PAUSED",null,null)
        _ui.value=_ui.value.copy(state=SessionState.PAUSED,boxes=emptyList(),activeTracks=0)
        sync.immediate()
    }
    suspend fun resume()=withContext(dispatcher) {
        if(_ui.value.state!=SessionState.PAUSED) return@withContext
        pausedDurationNs+=SystemClock.elapsedRealtimeNanos()-(pausedAtNs ?: SystemClock.elapsedRealtimeNanos())
        pausedAtNs=null; recorder.event("survey_state",obj("state" to j("RUNNING"))); saveSession("RUNNING",null,null)
        _ui.value=_ui.value.copy(state=SessionState.RUNNING)
    }
    suspend fun stop(reason:String="user_stopped")=withContext(dispatcher) {
        if(_ui.value.state !in setOf(SessionState.RUNNING,SessionState.PAUSED)) return@withContext
        _ui.value=_ui.value.copy(state=SessionState.STOPPING)
        try {
            tracker.flush().forEach { checkpoint(it,true,reason) }
            flushBatches()
            archiveFiles=recorder.finish(reason)
            saveSession(if(reason=="backgrounded") "INTERRUPTED" else "COMPLETED",System.currentTimeMillis(),reason)
            _ui.value=_ui.value.copy(state=SessionState.COMPLETED,boxes=emptyList(),activeTracks=0)
        } catch(e:Exception) {
            archiveFiles=recorder.finish("save_error")
            runCatching { saveSession("ERROR",System.currentTimeMillis(),"save_error") }
            _ui.value=_ui.value.copy(state=SessionState.ERROR,error="Could not finish saving: ${e.message}")
        }
        finally { sync.immediate() }
    }
    suspend fun videoStarted(path:String)=withContext(dispatcher) { videoFile=path; _ui.value=_ui.value.copy(videoPath=path) }
    suspend fun videoFinished(path:String,success:Boolean)=withContext(dispatcher) {
        if(path!=videoFile) return@withContext
        videoFinalized=success
        if(sessionId!=null) saveSession(lastSavedState,lastSavedEndMs,if(success)null else "video_encoder_failed")
        if(!success) _ui.value=_ui.value.copy(error="Local video could not be saved. Metadata collection is available.")
        sync.immediate()
    }
    suspend fun location(sample:LocationSample)=withContext(dispatcher) {
        if(_ui.value.state in setOf(SessionState.RUNNING,SessionState.PAUSED)) recorder.event("location_route_delivery",encoded(sample).jsonObject,sample.monotonic_ns)
        if(_ui.value.state!=SessionState.RUNNING || sample.id==lastLocationId || sample.monotonic_ns<startedNs) return@withContext
        lastLocationId=sample.id; route+=sample
        // locations arrive independently of inference
        if(SystemClock.elapsedRealtimeNanos()-lastBatchNs>=config.checkpointMs*1_000_000) {
            flushBatches(); lastBatchNs=SystemClock.elapsedRealtimeNanos(); sync.immediate()
        }
    }
    suspend fun process(frame:CameraFrame)=withContext(dispatcher) {
        if(_ui.value.state!=SessionState.RUNNING) return@withContext
        received++
        val realClock=frame.clockRealtime && abs(frame.sensorNs-frame.receiptNs)<10_000_000_000L
        val ns=if(realClock) frame.sensorNs else frame.receiptNs
        val power=context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val severe=Build.VERSION.SDK_INT>=29 && power.currentThermalStatus>=PowerManager.THERMAL_STATUS_SEVERE
        val fps=if(severe) min(2,config.targetFps) else config.targetFps
        if(ns<=lastProcessedNs || ns<startedNs || ns-lastProcessedNs<1_000_000_000L/fps) { skipped++; return@withContext }
        val begin=System.nanoTime(); lastProcessedNs=ns
        val time=FrameTime(++frameIndex,startedUtc+(ns-startedNs)/1_000_000,ns,frame.sensorNs,
            if(realClock) "sensor_realtime" else "callback_receipt_approximate",frame.rotation,frame.bitmap.width,frame.bitmap.height)
        try {
            recorder.event("analysis_frame",obj("frame" to encoded(time),"camera" to frame.cameraContext,
                "sensor_snapshot" to recorder.snapshot()),ns)
            val detected=detector.detect(frame.bitmap)
            latency.addLast(detected.inferenceMs); while(latency.size>128) latency.removeFirst()
            val trackingStart=System.nanoTime()
            val result=tracker.update(detected.targets,time)
            lastTrackingMs=(System.nanoTime()-trackingStart)/1e6
            val match=locations.buffer.closest(ns,config.gpsMaxAgeMs)
            val context=sceneContext(frame,detected)
            // every lamp box in the frame, for the cloud stage
            val frameLamps=JsonArray(result.matched.map { obj("track_id" to j(it.id),"observation_id" to j(it.observationId),"bbox" to encoded(it.detection.bbox),"score" to j(it.detection.score)) })
            val physicsContext=obj("frame_lamps" to frameLamps,
                "camera" to ((context["camera"] as? JsonObject)?.let { JsonObject(it-"all_characteristics") } ?: JsonNull),"device_sensors" to (context["device_sensors"] ?: JsonNull),"rotation_degrees" to (context["rotation_degrees"] ?: JsonNull),
                "frame_width" to (context["frame_width"] ?: JsonNull),"frame_height" to (context["frame_height"] ?: JsonNull))
            val samples=mutableListOf<TrackSampleEntity>()
            for(t in result.matched) {
                val sample=TrackSample(t.observationId,t.id,time,t.detection,match.sample,match.offsetMs,match.quality)
                recorder.event("track_sample",obj("sample" to encoded(sample)),ns)
                t.history.addLast(sample); while(t.history.size>config.maxHistory) t.history.removeFirst()
                samples+=TrackSampleEntity(sessionId=sessionId!!,observationId=t.observationId,monotonicNs=ns,utcMs=time.utc_ms,payload=encoded(sample).toString())
                if(ns-t.lastCloudSampleNs>=config.cloudTrackSampleMs*1_000_000) { cloudSamples+=sample; t.lastCloudSampleNs=ns }
                evidence.observe(t,frame.bitmap,time,context,match.sample,match.offsetMs,match.quality)
                if(savePhysicsFrames) physicsFrames.observe(t.id,time,t.detection.bbox,t.detection.score,physicsContext) { fullFrameJpeg(frame.bitmap) }
                seenTracks+=t.observationId
                if(ns-t.lastCheckpointNs>=config.checkpointMs*1_000_000) checkpoint(t,false,"live")
            }
            withContext(Dispatchers.IO) { records.sampleRows(samples) }
            result.completed.forEach { checkpoint(it,true,"lost_timeout") }
            if(ns-lastBatchNs>=config.checkpointMs*1_000_000) { flushBatches(); lastBatchNs=ns; sync.immediate() }
            if(ns-lastSessionNs>=5_000_000_000L) { saveSession("RUNNING",null,null); lastSessionNs=ns }
            val elapsed=max(1L,ns-startedNs-pausedDurationNs)
            lastPipelineMs=(System.nanoTime()-begin)/1e6
            consecutiveErrors=0
            _ui.value=_ui.value.copy(elapsedMs=(ns-startedNs)/1_000_000,tracks=seenTracks.size,activeTracks=result.active.size,
                fps=(frameIndex-errors)*1e9/elapsed,latencyMs=detected.inferenceMs,width=time.width,height=time.height,
                boxes=if(config.showOverlay) result.active.map { t -> OverlayBox(t.detection.bbox,"#${t.id}") } else emptyList())
        } catch(e:Exception) {
            errors++; consecutiveErrors++
            _ui.value=_ui.value.copy(error="Frame skipped: ${e.message}")
            if(consecutiveErrors>=10) {
                tracker.flush().forEach { t -> runCatching { checkpoint(t,true,"pipeline_error") } }
                runCatching { flushBatches(); archiveFiles=recorder.finish("pipeline_error"); saveSession("ERROR",System.currentTimeMillis(),"repeated_pipeline_failure") }
                _ui.value=_ui.value.copy(state=SessionState.ERROR,boxes=emptyList(),error="Repeated pipeline failure: ${e.message}")
                sync.immediate()
            }
        }
    }
    private fun versions()=obj("app" to j(BuildConfig.VERSION_NAME),"model" to j(detector.metadata.model_version),
        "model_sha256" to j(detector.metadata.sha256),"inference_runtime" to j(detector.runtimeId),"tracker" to j("centre-scale-hungarian-velocity-v1"),
        "configuration" to j(config.version))
    private fun sceneContext(frame:CameraFrame,detected:DetectionResult)=obj(
        "device" to obj("manufacturer" to j(Build.MANUFACTURER),"model" to j(Build.MODEL),"android_sdk" to j(Build.VERSION.SDK_INT)),
        "camera" to frame.cameraContext,"frame_width" to j(frame.bitmap.width),"frame_height" to j(frame.bitmap.height),
        "model_deployment" to encoded(detector.metadata),
        "rotation_degrees" to j(frame.rotation),"device_sensors" to recorder.snapshot(),
        "inference_ms" to j(detected.inferenceMs))
    private suspend fun checkpoint(t:Track,finished:Boolean,reason:String) {
        val selected=evidence.best(t.id)
        val trajectoryTail=t.history.fold(mutableListOf<TrackSample>()) { acc,s ->
                if(acc.isEmpty() || s.frame.monotonic_ns-acc.last().frame.monotonic_ns>=config.cloudTrackSampleMs*1_000_000) acc+=s
                acc
            }.takeLast(16)
        var localFile:String?=null; var cloudFile:String?=null; var imageHash:String?=null; var imageError:String?=null
        var contextLocal:String?=null; var contextCloud:String?=null; var contextHash:String?=null; var contextError:String?=null
        if(selected?.jpeg!=null && StatFs(context.filesDir.path).availableBytes>30L*1024*1024) {
            try {
                imageHash=MessageDigest.getInstance("SHA-256").digest(selected.jpeg).joinToString("") { "%02x".format(it) }
                val file=File(context.filesDir,"evidence/$sessionId/${t.observationId}-$imageHash.jpg")
                withContext(Dispatchers.IO) {
                    file.parentFile!!.mkdirs()
                    if(!file.exists()) { val temp=File(file.path+".tmp"); temp.writeBytes(selected.jpeg); check(temp.renameTo(file)) }
                }
                localFile=file.absolutePath; cloudFile="$ownerId/$sessionId/${file.name}"
            } catch(e:Exception) { imageError="image_write_failed"; localFile=null; cloudFile=null }
        } else if(selected?.jpeg!=null) imageError="low_storage"
        if(selected?.sceneJpeg!=null && StatFs(context.filesDir.path).availableBytes>30L*1024*1024) {
            try {
                contextHash=MessageDigest.getInstance("SHA-256").digest(selected.sceneJpeg).joinToString("") { "%02x".format(it) }
                val file=File(context.filesDir,"evidence/$sessionId/${t.observationId}-scene-$contextHash.jpg")
                withContext(Dispatchers.IO) {
                    file.parentFile!!.mkdirs()
                    if(!file.exists()) { val temp=File(file.path+".tmp"); temp.writeBytes(selected.sceneJpeg); check(temp.renameTo(file)) }
                }
                contextLocal=file.absolutePath; contextCloud="$ownerId/$sessionId/${file.name}"
            } catch(e:Exception) { contextError="image_write_failed" }
        } else if(selected?.sceneJpeg!=null) contextError="low_storage"
        if(finished && selected!=null) {
            val directory=File(context.filesDir,"surveys/$sessionId").apply { mkdirs() }
            val refs=mutableListOf<JsonElement>()
            for((kind,bytes) in listOf("lamp_crop" to selected.cropPng)) if(bytes!=null) {
                val hash=MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
                val file=File(directory,"$kind-$hash.png")
                withContext(Dispatchers.IO) { if(!file.exists()) file.writeBytes(bytes) }
                refs+=obj("kind" to j(kind),"file" to j(file.name),"sha256" to j(hash))
            }
            recorder.event("selected_evidence",obj("observation_id" to j(t.observationId),"track_id" to j(t.id),"frame" to encoded(selected.frame),
                "bbox" to encoded(selected.bbox),"crop_rect_in_frame_pixels" to encoded(selected.cropRect),"files" to JsonArray(refs),"scene_context" to selected.context))
        }
        // full-resolution frames at different distances for the cloud stage, sent when the track ends
        val extraRefs=mutableListOf<JsonElement>()
        if(contextLocal!=null && contextCloud!=null) extraRefs+=obj("local" to j(contextLocal),"cloud" to j(contextCloud))
        val physicsRefs=mutableListOf<JsonElement>(); var physicsError:String?=null
        if(finished && savePhysicsFrames) for(pf in physicsFrames.frames(t.id)) {
            if(StatFs(context.filesDir.path).availableBytes<=30L*1024*1024) { physicsError="low_storage"; break }
            try {
                val hash=MessageDigest.getInstance("SHA-256").digest(pf.jpeg).joinToString("") { "%02x".format(it) }
                val file=File(context.filesDir,"evidence/$sessionId/${t.observationId}-frame-$hash.jpg")
                withContext(Dispatchers.IO) {
                    file.parentFile!!.mkdirs()
                    if(!file.exists()) { val temp=File(file.path+".tmp"); temp.writeBytes(pf.jpeg); check(temp.renameTo(file)) }
                }
                val cloud="$ownerId/$sessionId/${file.name}"
                extraRefs+=obj("local" to j(file.absolutePath),"cloud" to j(cloud))
                physicsRefs+=obj("storage_bucket" to j("evidence"),"storage_path" to j(cloud),"sha256" to j(hash),"mime_type" to j("image/jpeg"),
                    "byte_count" to j(pf.jpeg.size),"frame" to encoded(pf.frame),"bbox" to encoded(pf.bbox),"detection_score" to j(pf.score),"context" to pf.context)
            } catch(e:Exception) { physicsError="image_write_failed" }
        }
        val extraImages=if(extraRefs.isEmpty()) null else JsonArray(extraRefs)
        val payload=obj("schema_version" to j(1),"observation_id" to j(t.observationId),"session_id" to j(sessionId),"track_id" to j(t.id),
            "finished" to j(finished),"completion_reason" to j(reason),
            "first_seen_utc_ms" to j(t.firstTime.utc_ms),"last_seen_utc_ms" to j(t.lastTime.utc_ms),
            "first_seen_monotonic_ns" to j(t.firstTime.monotonic_ns),"last_seen_monotonic_ns" to j(t.lastTime.monotonic_ns),
            "record_created_utc_ms" to j(System.currentTimeMillis()),"timing_basis" to j(t.lastTime.timing_basis),
            "device_observation_location" to (selected?.location?.let { encoded(it) } ?: JsonNull),
            "gps_time_offset_ms" to j(selected?.gpsOffsetMs),"gps_quality" to j(selected?.gpsQuality ?: "missing"),
            "lamp_location" to JsonNull,
            "detection_summary" to obj("count" to j(t.matches),"mean_score" to j(t.scoreSum/t.matches),"min_score" to j(t.minScore),"max_score" to j(t.maxScore),
                "class_name" to j(t.detection.class_name),"class_id" to j(t.detection.class_id),"source" to j(t.detection.source),"score_kind" to j(t.detection.score_kind),
                "representative_bbox" to encoded(selected?.bbox ?: t.detection.bbox),"bbox_convention" to j("XYXY_NORMALIZED_ROTATED_CROPPED_FRAME")),
            "track_summary" to obj("duration_ms" to j((t.lastTime.monotonic_ns-t.firstTime.monotonic_ns)/1_000_000),"matched_samples" to j(t.matches),
                "first_bbox" to encoded(t.firstBox),"final_bbox" to encoded(t.detection.bbox),"max_area_bbox" to encoded(t.maxAreaBox),
                "trajectory_tail" to encoded(trajectoryTail),"tail_is_truncated" to j(t.matches>trajectoryTail.size),"separate_cloud_sample_interval_ms" to j(config.cloudTrackSampleMs)),
            "scene_context" to (selected?.context ?: JsonNull),
            "evidence" to obj("storage_bucket" to j("evidence"),"storage_path" to j(cloudFile),"sha256" to j(imageHash),"mime_type" to j("image/jpeg"),
                "byte_count" to j(if(cloudFile!=null) selected?.jpeg?.size else null),"selection_frame" to (selected?.frame?.let { encoded(it) } ?: JsonNull),
                "crop_rect_in_frame_pixels" to (selected?.cropRect?.let { encoded(it) } ?: JsonNull),"selection_score" to j(selected?.score),"missing_reason" to j(imageError)),
            "scene_context_image" to obj("storage_bucket" to j("evidence"),"storage_path" to j(contextCloud),"sha256" to j(contextHash),
                "mime_type" to j("image/jpeg"),"byte_count" to j(if(contextCloud!=null) selected?.sceneJpeg?.size else null),
                "source_frame" to (selected?.frame?.let { encoded(it) } ?: JsonNull),"missing_reason" to j(contextError)),
            "physics_frames" to obj("schema_version" to j(1),"frames" to JsonArray(physicsRefs),"missing_reason" to j(physicsError),
                "bbox_convention" to j("XYXY_NORMALIZED_ROTATED_CROPPED_FRAME"),"selection" to j("up to 5 per track, spread over box size, full resolution JPEG q90")),
            "algorithm_versions" to versions())
        t.revision++
        withContext(Dispatchers.IO) { records.observation(ObservationEntity(t.observationId,sessionId!!,t.id,finished,t.revision,payload.toString()),ownerId,localFile,cloudFile,extraImages?.toString()) }
        t.lastCheckpointNs=t.lastTime.monotonic_ns
        if(finished) { evidence.discard(t.id); physicsFrames.discard(t.id) }
    }
    private suspend fun flushBatches() {
        val sid=sessionId ?: return
        if(route.isNotEmpty()) {
            val copy=route.toList()
            val payload=obj("schema_version" to j(1),"session_id" to j(sid),"locations" to encoded(copy))
            withContext(Dispatchers.IO) { records.route(copy.map { LocationEntity(it.id,sid,it.utc_ms,encoded(it).toString()) },
                OutboxEntity(UUID.randomUUID().toString(),sid,ownerId,"LOCATION_BATCH",1,payload.toString())) }
            route.clear()
        }
        if(cloudSamples.isNotEmpty()) {
            val copy=cloudSamples.toList()
            val payload=obj("schema_version" to j(1),"session_id" to j(sid),"sampling_interval_ms" to j(config.cloudTrackSampleMs),"samples" to encoded(copy))
            withContext(Dispatchers.IO) { records.dao.enqueue(OutboxEntity(UUID.randomUUID().toString(),sid,ownerId,"TRACK_BATCH",1,payload.toString())) }
            cloudSamples.clear()
        }
    }
    private suspend fun saveSession(state:String,endMs:Long?,reason:String?) {
        val sid=sessionId ?: return
        val metrics=obj("analysis_delivered_frames" to j(received),"attempted_inference_frames" to j(frameIndex),"processed_frames" to j(max(0L,frameIndex-errors)),"rate_limited_frames" to j(skipped),"failed_frames" to j(errors),
            "camera_internal_drops" to JsonNull,"mean_inference_ms" to j(latency.takeIf { it.isNotEmpty() }?.average()),
            "p95_inference_ms" to j(latency.sorted().takeIf { it.isNotEmpty() }?.let { it[((it.size-1)*0.95).toInt()] }),
            "last_tracking_ms" to j(lastTrackingMs),"last_pipeline_ms" to j(lastPipelineMs))
        val archiveRefs=JsonArray(archiveFiles.filter { it.name=="manifest.json" || it.extension=="gz" }.map { file -> obj("name" to j(file.name),"storage_path" to j("$ownerId/$sid/survey/${file.name}"),
            "byte_count" to j(file.length()),"sha256" to j(SurveyRecorder.sha256(file))) })
        val extras=if(archiveFiles.isEmpty()) null else JsonArray(archiveFiles.map { file -> obj("local" to j(file.absolutePath),"cloud" to j("$ownerId/$sid/survey/${file.name}")) }).toString()
        val payload=obj("schema_version" to j(1),"session_id" to j(sid),"state" to j(state),"start_utc_ms" to j(startedUtc),"end_utc_ms" to j(endMs),
            "start_monotonic_ns" to j(startedNs),"completion_reason" to j(reason),"tracks" to j(seenTracks.size),
            "collection_mode" to j(if(config.allowMissingGps) "research_gps_override" else "field"),
            "runtime_configuration" to encoded(config),"model_deployment" to encoded(detector.metadata),"algorithm_versions" to versions(),"diagnostics" to metrics,
            "survey_archive" to obj("storage_bucket" to j("evidence"),"files" to archiveRefs,"telemetry_format" to j("gzip_json_lines"),"total_files" to j(archiveFiles.size),
                "contains_complete_processed_track_history" to j(true),"available_after_stop" to j(archiveFiles.isNotEmpty())),
            "video_saved_locally" to j(videoFile!=null),"video_finalized" to (videoFinalized?.let { j(it) } ?: JsonNull),"video_uploaded" to j(false))
        sessionRevision++
        withContext(Dispatchers.IO) { records.session(SessionEntity(sid,ownerId,startedUtc,endMs,state,sessionRevision,payload.toString()),extras) }
        lastSavedState=state; lastSavedEndMs=endMs
    }
}
