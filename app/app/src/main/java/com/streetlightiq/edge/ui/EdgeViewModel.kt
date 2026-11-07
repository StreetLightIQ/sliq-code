package com.streetlightiq.edge.ui

import android.content.Context
import android.os.SystemClock
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.streetlightiq.edge.camera.CameraController
import com.streetlightiq.edge.config.*
import com.streetlightiq.edge.data.local.RecordRepository
import com.streetlightiq.edge.data.remote.SupabaseCloud
import com.streetlightiq.edge.domain.*
import com.streetlightiq.edge.location.LocationProvider
import com.streetlightiq.edge.pipeline.EdgePipeline
import com.streetlightiq.edge.sync.*
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.io.File
import javax.inject.Inject

@HiltViewModel
class EdgeViewModel @Inject constructor(@ApplicationContext private val context:Context,
    val pipeline:EdgePipeline,val camera:CameraController,val locations:LocationProvider,
    private val preferences:Preferences,private val cloud:SupabaseCloud,
    private val records:RecordRepository,private val sync:SyncScheduler,private val engine:SyncEngine):ViewModel() {
    val ui=pipeline.ui
    val settings=preferences.settings.stateIn(viewModelScope,SharingStarted.Eagerly,Settings())
    val pending=records.dao.pendingCount().stateIn(viewModelScope,SharingStarted.Eagerly,0)
    val failed=records.dao.failedCount().stateIn(viewModelScope,SharingStarted.Eagerly,0)
    val sessions=records.dao.sessions().stateIn(viewModelScope,SharingStarted.Eagerly,emptyList())
    val cameraReady=MutableStateFlow(false)
    val operationMessage=MutableStateFlow<String?>(null)
    val signedInEmail=MutableStateFlow(cloud.vault.read()?.email ?: "")
    val busy=MutableStateFlow(false)
    val gpsLabel=MutableStateFlow("Waiting for GPS")
    val syncLabel=MutableStateFlow("")
    private val durableScope=CoroutineScope(SupervisorJob()+Dispatchers.Main.immediate)
    init {
        camera.onFrame={ pipeline.process(it) }
        camera.onCapture={ pipeline.capture(it) }
        (locations as? com.streetlightiq.edge.location.AndroidLocationProvider)?.onRawSample={ pipeline.rawLocation(it) }
        camera.onReady={ cameraReady.value=it }
        camera.onError={ message ->
            operationMessage.value=message
            if(message.startsWith("Camera error")) stop("camera_error")
        }
        camera.onVideoFinalized={ path,success -> durableScope.launch { pipeline.videoFinished(path,success) } }
        viewModelScope.launch { pipeline.initialize() }
        viewModelScope.launch { locations.latest.filterNotNull().collect { pipeline.location(it) } }
        viewModelScope.launch {
            while(isActive) {
                val sample=locations.latest.value
                gpsLabel.value=when {
                    sample==null -> "Waiting for GPS"
                    (SystemClock.elapsedRealtimeNanos()-sample.monotonic_ns)>5_000_000_000L -> "GPS stale"
                    sample.is_mock -> "Mock location · research only"
                    else -> "GPS ±${sample.accuracy_m.toInt()} m"
                }
                syncLabel.value=engine.lastMessage
                if(ui.value.state==SessionState.RUNNING && pending.value>0) sync.immediate()
                delay(2000)
            }
        }
    }
    fun beginLocation() {
        runCatching { locations.start() }.onFailure { operationMessage.value=it.message }
    }
    fun start()=viewModelScope.launch {
        operationMessage.value=null
        pipeline.start(cameraReady.value)
        if(ui.value.state==SessionState.RUNNING && settings.value.runtime.saveLocalVideo) {
            val file=File(context.filesDir,"video/${ui.value.sessionId}.mp4")
            try { camera.startVideo(file); pipeline.videoStarted(file.absolutePath) }
            catch(e:Exception) { operationMessage.value=e.message; pipeline.stop("video_start_failed") }
        }
    }
    fun stop(reason:String="user_stopped") {
        // app scope, so Room writes finish even if the ViewModel is cleared
        durableScope.launch { camera.stopVideo(); pipeline.stop(reason) }
    }
    fun pauseOrResume()=viewModelScope.launch {
        if(ui.value.state==SessionState.RUNNING) { camera.stopVideo(); pipeline.pause() }
        else if(ui.value.state==SessionState.PAUSED) {
            pipeline.resume()
            if(settings.value.runtime.saveLocalVideo) {
                val file=File(context.filesDir,"video/${ui.value.sessionId}-${System.currentTimeMillis()}.mp4")
                runCatching { camera.startVideo(file) }.onSuccess { pipeline.videoStarted(file.absolutePath) }
                    .onFailure { operationMessage.value=it.message }
            }
        }
    }
    fun backgrounded() { stop("backgrounded"); locations.stop() }
    fun save(settings:Settings,email:String,password:String,login:Boolean)=viewModelScope.launch {
        if(busy.value) return@launch
        busy.value=true
        try {
            require(ui.value.state !in setOf(SessionState.RUNNING,SessionState.PAUSED)) { "Stop collection before changing settings." }
            val next=settings.copy(backend=settings.backend.copy(url=settings.backend.url.trim().trimEnd('/'),publicKey=settings.backend.publicKey.trim()))
            val old=preferences.get()
            require(pending.value==0 || next.backend.url==old.backend.url) { "Upload the pending records before changing the backend project." }
            if(login) {
                val prior=cloud.vault.read()
                val signed=cloud.signIn(next.backend,email,password)
                if(pending.value>0 && prior!=null && prior.userId!=signed.userId) {
                    cloud.vault.save(prior); error("Pending records belong to the previous account. Sign in to that account first.")
                }
                signedInEmail.value=signed.email
            }
            preferences.save(next)
            operationMessage.value=if(login) "Signed in. Wait for camera and GPS, then start." else "Settings saved."
            sync.immediate()
        } catch(e:Exception) { operationMessage.value=e.message }
        finally { busy.value=false }
    }
    fun retry()=viewModelScope.launch { records.dao.retryFailures(); sync.immediate(); operationMessage.value="Retry scheduled." }
}
