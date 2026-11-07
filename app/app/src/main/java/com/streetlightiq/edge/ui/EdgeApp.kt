package com.streetlightiq.edge.ui

import android.Manifest
import android.app.Activity
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.hardware.display.DisplayManager
import android.os.Handler
import android.os.Looper
import android.view.Surface as DisplaySurface
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.streetlightiq.edge.config.*
import com.streetlightiq.edge.domain.SessionState
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.min

// Night theme: dark surfaces (less glare in a car at night), streetlight amber accent.
private val Amber=Color(0xFFFFC14D)
private val Good=Color(0xFF5FD39A)
private val Warn=Color(0xFFFFB347)
private val Muted=Color(0xFF9AA4B2)
private val colors=darkColorScheme(primary=Amber,onPrimary=Color(0xFF231800),secondary=Color(0xFF8FB8D0),
    background=Color(0xFF0D1117),surface=Color(0xFF151B23),surfaceVariant=Color(0xFF1E2632),onSurface=Color(0xFFE6E9EE),
    onSurfaceVariant=Color(0xFFC3CAD4),outline=Color(0xFF3A4452),error=Color(0xFFFF8A80))

@Composable
fun EdgeApp(model:EdgeViewModel,keepAwake:(Boolean)->Unit) {
    val ui by model.ui.collectAsStateWithLifecycle()
    val settings by model.settings.collectAsStateWithLifecycle()
    val pending by model.pending.collectAsStateWithLifecycle()
    val failed by model.failed.collectAsStateWithLifecycle()
    val ready by model.cameraReady.collectAsStateWithLifecycle()
    val operationBusy by model.busy.collectAsStateWithLifecycle()
    val gps by model.gpsLabel.collectAsStateWithLifecycle()
    val sync by model.syncLabel.collectAsStateWithLifecycle()
    val email by model.signedInEmail.collectAsStateWithLifecycle()
    val message by model.operationMessage.collectAsStateWithLifecycle()
    val sessions by model.sessions.collectAsStateWithLifecycle()
    val context=LocalContext.current
    val view=LocalView.current
    val activity=context as? Activity
    var page by remember { mutableStateOf("collect") }
    var permissionVersion by remember { mutableIntStateOf(0) }
    var showRotateHint by remember { mutableStateOf(false) }
    val permissions=rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissionVersion++; model.beginLocation() }
    val cameraPermission=remember(permissionVersion) { ContextCompat.checkSelfPermission(context,Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED }
    val locationPermission=remember(permissionVersion) { ContextCompat.checkSelfPermission(context,Manifest.permission.ACCESS_FINE_LOCATION)==PackageManager.PERMISSION_GRANTED }
    val active=ui.state in setOf(SessionState.RUNNING,SessionState.PAUSED,SessionState.STOPPING)

    // Track the screen rotation, including 90 <-> 270 flips, which do not change the configuration.
    var rotation by remember { mutableIntStateOf(view.display?.rotation ?: DisplaySurface.ROTATION_0) }
    DisposableEffect(view) {
        val displays=context.getSystemService(android.content.Context.DISPLAY_SERVICE) as android.hardware.display.DisplayManager
        val listener=object:android.hardware.display.DisplayManager.DisplayListener {
            override fun onDisplayAdded(id:Int) {}
            override fun onDisplayRemoved(id:Int) {}
            override fun onDisplayChanged(id:Int) { view.display?.let { if(it.displayId==id) rotation=it.rotation } }
        }
        displays.registerDisplayListener(listener,Handler(Looper.getMainLooper()))
        onDispose { displays.unregisterDisplayListener(listener) }
    }
    // Configuration changes (portrait <-> landscape) arrive here reliably; the display listener covers 90 <-> 270 flips.
    val configuration=LocalConfiguration.current
    LaunchedEffect(configuration.orientation,configuration.screenWidthDp) { view.display?.let { rotation=it.rotation } }
    val landscape=configuration.orientation==Configuration.ORIENTATION_LANDSCAPE
    LaunchedEffect(rotation) { model.camera.setRotation(rotation) }
    LaunchedEffect(landscape) { if(landscape) showRotateHint=false }

    // The detector expects landscape frames: lock the screen to the current landscape side while collecting, free it otherwise.
    LaunchedEffect(active) {
        keepAwake(active)
        activity?.requestedOrientation=when {
            !active -> ActivityInfo.SCREEN_ORIENTATION_FULL_SENSOR
            rotation==DisplaySurface.ROTATION_270 -> ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE
            else -> ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        }
    }
    DisposableEffect(Unit) { onDispose { keepAwake(false) } }
    // Full-screen camera in landscape; normal system bars for setup.
    LaunchedEffect(landscape,page) {
        activity?.window?.let { window ->
            WindowCompat.getInsetsController(window,view).apply {
                if(landscape && page=="collect") { systemBarsBehavior=WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE; hide(WindowInsetsCompat.Type.systemBars()) }
                else show(WindowInsetsCompat.Type.systemBars())
            }
        }
    }

    fun startSurvey() {
        if(!landscape) { showRotateHint=true; return }
        model.camera.setRotation(rotation)
        model.start()
    }
    val canStart=!operationBusy && ui.modelReady && ready && email.isNotBlank()
    val elapsed="${ui.elapsedMs/60000}:${(ui.elapsedMs/1000%60).toString().padStart(2,'0')}"
    // Always say why Start is unavailable (amber), then any error (red) or status message (grey).
    val blocker=if(active) null else when {
        !cameraPermission || !locationPermission -> null
        email.isBlank() -> "Start needs a backend sign-in: open Settings, enter the Supabase URL, key, email and password, then tap Save and sign in."
        !ui.modelReady -> "Loading the detector…"
        !ready -> "Waiting for the camera…"
        operationBusy -> "Busy: wait a moment."
        !gps.contains("±") && !settings.runtime.allowMissingGps -> "No recent GPS fix yet. Go outdoors with a clear sky; Start will refuse without it."
        else -> null
    }
    val notices=buildList {
        blocker?.let { add(it to Warn) }
        ui.error?.let { add(it to colors.error) }
        message?.takeIf { it!=ui.error }?.let { add(it to Muted) }
        if(ui.state==SessionState.COMPLETED) add("Session saved. Uploads continue in the background." to Muted)
    }

    MaterialTheme(colorScheme=colors) {
        Surface(Modifier.fillMaxSize(),color=colors.background) {
            if(page=="settings") {
                Column(Modifier.fillMaxSize().safeDrawingPadding().padding(horizontal=20.dp,vertical=12.dp)) {
                    Row(Modifier.fillMaxWidth(),verticalAlignment=Alignment.CenterVertically) {
                        Text("Settings",style=MaterialTheme.typography.titleLarge,fontWeight=FontWeight.Bold,modifier=Modifier.weight(1f))
                        TextButton(onClick={ page="collect" }) { Text("Done") }
                    }
                    Spacer(Modifier.height(8.dp))
                    SettingsScreen(model,settings,email,pending,failed,sync,message)
                }
            } else if(landscape) {
                // Survey layout: camera fills the height, compact control panel on the side.
                Row(Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.displayCutout).padding(10.dp),horizontalArrangement=Arrangement.spacedBy(12.dp)) {
                    Box(Modifier.weight(1f).fillMaxHeight(),contentAlignment=Alignment.Center) {
                        if(cameraPermission) Box(Modifier.aspectRatio(16f/9f).clip(RoundedCornerShape(14.dp))) {
                            CameraPanel(model,settings,Modifier.fillMaxSize())
                            if(active) RecordingPill(ui.state==SessionState.PAUSED,elapsed,Modifier.align(Alignment.TopStart).padding(10.dp))
                        } else NoCamera(Modifier.fillMaxSize())
                    }
                    Column(Modifier.width(210.dp).fillMaxHeight().verticalScroll(rememberScrollState()),verticalArrangement=Arrangement.spacedBy(10.dp)) {
                        Row(verticalAlignment=Alignment.CenterVertically) {
                            Text("StreetLightIQ",style=MaterialTheme.typography.titleMedium,fontWeight=FontWeight.Bold,modifier=Modifier.weight(1f))
                            if(!active) TextButton(onClick={ page="settings" },contentPadding=PaddingValues(horizontal=8.dp)) { Text("Settings") }
                        }
                        StatusChip("Model",ui.modelReady)
                        StatusChip("Camera",ready)
                        StatusChip(gps,locationPermission && gps.contains("±"))
                        StatsCard(ui.tracks,elapsed,pending,stacked=false)
                        notices.forEach { (text,color) -> Text(text,style=MaterialTheme.typography.bodySmall,color=color,maxLines=5,overflow=TextOverflow.Ellipsis) }
                        Spacer(Modifier.weight(1f))
                        Controls(active,ui.state,canStart,cameraPermission && locationPermission,
                            grant={ permissions.launch(arrayOf(Manifest.permission.CAMERA,Manifest.permission.ACCESS_FINE_LOCATION,Manifest.permission.ACCESS_COARSE_LOCATION)) },
                            start=::startSurvey,pause={ model.pauseOrResume() },stop={ model.stop() },vertical=true)
                    }
                }
            } else {
                // Setup layout (portrait): settings, preview and readiness. Surveys start in landscape.
                Column(Modifier.fillMaxSize().safeDrawingPadding().padding(horizontal=20.dp,vertical=12.dp),verticalArrangement=Arrangement.spacedBy(14.dp)) {
                    Row(Modifier.fillMaxWidth(),verticalAlignment=Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text("StreetLightIQ",style=MaterialTheme.typography.headlineSmall,fontWeight=FontWeight.Bold)
                            Text("Night streetlight survey",style=MaterialTheme.typography.bodySmall,color=Amber)
                        }
                        OutlinedButton(enabled=!active,onClick={ page="settings" }) { Text("Settings") }
                    }
                    Box(Modifier.fillMaxWidth().aspectRatio(16f/9f).clip(RoundedCornerShape(16.dp))) {
                        if(cameraPermission) CameraPanel(model,settings,Modifier.fillMaxSize()) else NoCamera(Modifier.fillMaxSize())
                    }
                    Row(horizontalArrangement=Arrangement.spacedBy(8.dp)) {
                        StatusChip("Model",ui.modelReady); StatusChip("Camera",ready); StatusChip(gps,locationPermission && gps.contains("±"))
                    }
                    StatsCard(ui.tracks,elapsed,pending,stacked=true)
                    notices.forEach { (text,color) -> Text(text,style=MaterialTheme.typography.bodySmall,color=color) }
                    Spacer(Modifier.weight(1f))
                    RotateBanner()
                    Controls(active,ui.state,canStart,cameraPermission && locationPermission,
                        grant={ permissions.launch(arrayOf(Manifest.permission.CAMERA,Manifest.permission.ACCESS_FINE_LOCATION,Manifest.permission.ACCESS_COARSE_LOCATION)) },
                        start=::startSurvey,pause={ model.pauseOrResume() },stop={ model.stop() },vertical=false)
                    if(sessions.isNotEmpty() && !active) {
                        val last=sessions.first()
                        Text("Last survey: ${SimpleDateFormat("dd MMM HH:mm",Locale.US).format(Date(last.startedMs))} · ${last.state}",style=MaterialTheme.typography.bodySmall,color=Muted)
                    }
                }
            }
            if(showRotateHint) AlertDialog(onDismissRequest={ showRotateHint=false },
                confirmButton={ TextButton(onClick={ showRotateHint=false }) { Text("OK") } },
                title={ Text("Turn your phone sideways") },
                text={ Text("Surveys run in landscape only, so the camera sees the road the way the detector was trained. Rotate the phone, mount it with the rear camera facing the road, then tap Start. The screen stays locked in landscape until you stop.") })
        }
    }
}

@Composable private fun NoCamera(modifier:Modifier) {
    Box(modifier.background(colors.surfaceVariant),contentAlignment=Alignment.Center) {
        Text("Camera access needed for the preview",style=MaterialTheme.typography.bodyMedium,color=Muted,textAlign=TextAlign.Center,modifier=Modifier.padding(16.dp))
    }
}

@Composable private fun StatusChip(label:String,ok:Boolean) {
    Surface(shape=RoundedCornerShape(50),color=colors.surfaceVariant) {
        Row(Modifier.padding(horizontal=10.dp,vertical=6.dp),verticalAlignment=Alignment.CenterVertically) {
            Box(Modifier.size(8.dp).clip(CircleShape).background(if(ok) Good else Warn))
            Spacer(Modifier.width(6.dp))
            Text(label,style=MaterialTheme.typography.labelMedium,maxLines=1)
        }
    }
}

@Composable private fun StatsCard(tracks:Int,elapsed:String,pending:Int,stacked:Boolean) {
    Surface(shape=RoundedCornerShape(14.dp),color=colors.surface,modifier=Modifier.fillMaxWidth()) {
        Row(Modifier.padding(horizontal=14.dp,vertical=12.dp),horizontalArrangement=Arrangement.SpaceBetween) {
            Stat("$tracks","tracks",Modifier.weight(1f))
            Stat(elapsed,"time",Modifier.weight(1f))
            Stat("$pending",if(stacked) "to upload" else "queued",Modifier.weight(1f))
        }
    }
}

@Composable private fun Stat(value:String,label:String,modifier:Modifier) {
    Column(modifier,horizontalAlignment=Alignment.CenterHorizontally) {
        Text(value,style=MaterialTheme.typography.titleLarge,fontWeight=FontWeight.SemiBold,maxLines=1)
        Text(label,style=MaterialTheme.typography.labelSmall,color=Muted,maxLines=1)
    }
}

@Composable private fun RecordingPill(paused:Boolean,elapsed:String,modifier:Modifier) {
    Surface(shape=RoundedCornerShape(50),color=Color(0xB3000000),modifier=modifier) {
        Row(Modifier.padding(horizontal=10.dp,vertical=5.dp),verticalAlignment=Alignment.CenterVertically) {
            Box(Modifier.size(8.dp).clip(CircleShape).background(if(paused) Warn else Color(0xFFFF5252)))
            Spacer(Modifier.width(6.dp))
            Text(if(paused) "PAUSED  $elapsed" else "REC  $elapsed",style=MaterialTheme.typography.labelMedium,color=Color.White)
        }
    }
}

@Composable private fun RotateBanner() {
    Surface(shape=RoundedCornerShape(14.dp),color=Color(0x26FFC14D),modifier=Modifier.fillMaxWidth()) {
        Row(Modifier.padding(14.dp),verticalAlignment=Alignment.CenterVertically) {
            Text("⟳",style=MaterialTheme.typography.titleLarge,color=Amber)
            Spacer(Modifier.width(12.dp))
            Text("Turn the phone sideways (landscape) to start a survey.",style=MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable private fun Controls(active:Boolean,state:SessionState,canStart:Boolean,permitted:Boolean,grant:()->Unit,start:()->Unit,pause:()->Unit,stop:()->Unit,vertical:Boolean) {
    val big=Modifier.fillMaxWidth().height(52.dp)
    when {
        !permitted -> Button(onClick=grant,modifier=big) { Text("Grant camera and location") }
        active -> {
            val pauseButton=@Composable { m:Modifier -> OutlinedButton(onClick=pause,enabled=state!=SessionState.STOPPING,modifier=m) { Text(if(state==SessionState.PAUSED) "Resume" else "Pause") } }
            val stopButton=@Composable { m:Modifier -> Button(onClick=stop,enabled=state!=SessionState.STOPPING,modifier=m,colors=ButtonDefaults.buttonColors(containerColor=Color(0xFFE5534B),contentColor=Color.White)) { Text(if(state==SessionState.STOPPING) "Saving…" else "Stop") } }
            if(vertical) Column(verticalArrangement=Arrangement.spacedBy(8.dp)) { pauseButton(big); stopButton(big) }
            else Row(horizontalArrangement=Arrangement.spacedBy(10.dp)) { pauseButton(Modifier.weight(1f).height(52.dp)); stopButton(Modifier.weight(1f).height(52.dp)) }
        }
        else -> Button(onClick=start,enabled=canStart,modifier=big) { Text("Start survey",fontWeight=FontWeight.SemiBold) }
    }
}

@Composable private fun CameraPanel(model:EdgeViewModel,settings:Settings,modifier:Modifier) {
    val ui by model.ui.collectAsStateWithLifecycle()
    val owner=LocalLifecycleOwner.current
    var preview by remember { mutableStateOf<PreviewView?>(null) }
    Box(modifier.background(Color.Black).clipToBounds()) {
        AndroidView(factory={ context -> PreviewView(context).also { preview=it } },modifier=Modifier.fillMaxSize().clipToBounds())
        if(ui.width>0 && ui.height>0) Canvas(Modifier.fillMaxSize()) {
            val scale=min(size.width/ui.width,size.height/ui.height)
            val dx=(size.width-ui.width*scale)/2; val dy=(size.height-ui.height*scale)/2
            for(o in ui.boxes) {
                drawRect(Color(0xFF64EAB8),
                    Offset(dx+o.box.x1*ui.width*scale,dy+o.box.y1*ui.height*scale),
                    Size(o.box.width*ui.width*scale,o.box.height*ui.height*scale),style=Stroke(2.dp.toPx()))
                drawIntoCanvas { canvas -> canvas.nativeCanvas.drawText(o.label,dx+o.box.x1*ui.width*scale,
                    (dy+o.box.y1*ui.height*scale-6).coerceAtLeast(24f),android.graphics.Paint().apply {
                        color=android.graphics.Color.rgb(100,234,184)
                        textSize=14.dp.toPx(); isAntiAlias=true
                    }) }
            }
        }
    }
    val videoOn=settings.runtime.saveLocalVideo
    LaunchedEffect(preview,videoOn) { preview?.let { view -> view.post { model.camera.bind(owner,view,videoOn) } } }
    DisposableEffect(Unit) { onDispose { model.camera.unbind() } }
}

@Composable private fun SettingsScreen(model:EdgeViewModel,settings:Settings,email:String,pending:Int,failed:Int,sync:String,message:String?) {
    val busy by model.busy.collectAsStateWithLifecycle()
    var url by remember(settings.backend.url) { mutableStateOf(settings.backend.url) }
    var key by remember(settings.backend.publicKey) { mutableStateOf(settings.backend.publicKey) }
    var loginEmail by remember(email) { mutableStateOf(email) }
    var password by remember { mutableStateOf("") }
    var evidence by remember(settings.runtime.saveEvidence) { mutableStateOf(settings.runtime.saveEvidence) }
    var contextFrames by remember(settings.runtime.saveContextFrames) { mutableStateOf(settings.runtime.saveContextFrames) }
    var video by remember(settings.runtime.saveLocalVideo) { mutableStateOf(settings.runtime.saveLocalVideo) }
    var overlays by remember(settings.runtime.showOverlay) { mutableStateOf(settings.runtime.showOverlay) }
    var debugGps by remember(settings.runtime.allowMissingGps) { mutableStateOf(settings.runtime.allowMissingGps) }
    val ui by model.ui.collectAsStateWithLifecycle()
    val detectorLabel by model.pipeline.modelLabel.collectAsStateWithLifecycle()
    val optionalPermissions=rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { }
    fun configured()=settings.copy(backend=BackendConfig(url,key),runtime=settings.runtime.copy(saveEvidence=evidence,saveContextFrames=contextFrames,saveLocalVideo=video,showOverlay=overlays,allowMissingGps=debugGps))
    Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState()),verticalArrangement=Arrangement.spacedBy(10.dp)) {
        Text("Backend connection",fontWeight=FontWeight.Bold)
        Text("Run backend/setup.sql in Supabase, create a survey user, then enter the project URL and publishable key.",style=MaterialTheme.typography.bodySmall)
        OutlinedTextField(url,{url=it},label={Text("https://PROJECT.supabase.co")},singleLine=true,modifier=Modifier.fillMaxWidth())
        OutlinedTextField(key,{key=it},label={Text("Publishable / anon key")},singleLine=true,modifier=Modifier.fillMaxWidth())
        OutlinedTextField(loginEmail,{loginEmail=it},label={Text("Survey email")},singleLine=true,modifier=Modifier.fillMaxWidth())
        OutlinedTextField(password,{password=it},label={Text("Password")},visualTransformation=PasswordVisualTransformation(),singleLine=true,modifier=Modifier.fillMaxWidth())
        Button(enabled=!busy,onClick={ model.save(configured(),loginEmail,password,true); password="" },modifier=Modifier.fillMaxWidth()) { Text(if(busy) "Connecting…" else "Save and sign in") }
        if(email.isNotBlank()) Text("Signed in: $email",style=MaterialTheme.typography.bodySmall)
        HorizontalDivider()
        Text("Detector and sensors",fontWeight=FontWeight.Bold)
        Text(detectorLabel,style=MaterialTheme.typography.bodySmall)
        OutlinedButton(onClick={ optionalPermissions.launch(if(android.os.Build.VERSION.SDK_INT>=29) arrayOf(Manifest.permission.ACTIVITY_RECOGNITION,Manifest.permission.BODY_SENSORS) else arrayOf(Manifest.permission.BODY_SENSORS)) },modifier=Modifier.fillMaxWidth()) { Text("Allow optional motion / body sensors") }
        Text("Available sensors, GNSS and camera metadata are recorded during each survey. Unsupported or denied sensors appear in the inventory. Survey archive uploads after Stop.",style=MaterialTheme.typography.bodySmall)
        HorizontalDivider()
        Text("Collection",fontWeight=FontWeight.Bold)
        Toggle("Save representative crops",evidence) { evidence=it }
        Toggle("Save selected scene frames",contextFrames) { contextFrames=it }
        Toggle("Save video locally (no audio)",video) { video=it }
        Toggle("Show lamp boxes",overlays) { overlays=it }
        Toggle("Research: permit missing / mock GPS",debugGps) { debugGps=it }
        if(debugGps) Text("Override sessions are labelled research data. Missing coordinates stay null.",style=MaterialTheme.typography.bodySmall,color=MaterialTheme.colorScheme.error)
        OutlinedButton(onClick={model.save(configured(),loginEmail,"",false)},enabled=!busy,modifier=Modifier.fillMaxWidth()) { Text("Save collection settings") }
        HorizontalDivider()
        Text("$pending queued · $failed require attention",fontWeight=FontWeight.Bold)
        Text(sync,style=MaterialTheme.typography.bodySmall)
        OutlinedButton(onClick={model.retry()},modifier=Modifier.fillMaxWidth()) { Text("Retry uploads") }
        Text("Processing: ${"%.1f".format(ui.fps)} FPS · ${ui.latencyMs.toInt()} ms inference\nTracker: centre + scale + velocity / Hungarian\nDetector: $detectorLabel",style=MaterialTheme.typography.bodySmall)
        message?.let { Text(it,style=MaterialTheme.typography.bodySmall) }
    }
}
@Composable private fun Toggle(label:String,value:Boolean,onChange:(Boolean)->Unit) {
    Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.SpaceBetween) {
        Text(label,Modifier.weight(1f).padding(top=12.dp),style=MaterialTheme.typography.bodyMedium)
        Switch(value,onCheckedChange=onChange)
    }
}
