package com.streetlightiq.edge.location

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Looper
import androidx.core.content.ContextCompat
import com.google.android.gms.location.*
import com.streetlightiq.edge.domain.LocationSample
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.util.UUID
import kotlin.math.abs

data class LocationMatch(val sample:LocationSample?,val offsetMs:Long?,val quality:String)
class LocationBuffer(private val maxSamples:Int=600) {
    private val samples=ArrayDeque<LocationSample>()
    @Synchronized fun add(sample:LocationSample) {
        if(samples.any { it.monotonic_ns==sample.monotonic_ns }) return
        samples.addLast(sample); while(samples.size>maxSamples) samples.removeFirst()
    }
    @Synchronized fun closest(timeNs:Long,maxAgeMs:Long):LocationMatch {
        val sample=samples.minByOrNull { abs(it.monotonic_ns-timeNs) } ?: return LocationMatch(null,null,"missing")
        val offset=(sample.monotonic_ns-timeNs)/1_000_000
        if(abs(offset)>maxAgeMs) return LocationMatch(null,offset,"stale")
        return LocationMatch(sample,offset,if(sample.is_mock) "mock" else if(sample.accuracy_m<=20) "good" else "low_accuracy")
    }
    @Synchronized fun clear() { samples.clear() }
}
interface LocationProvider {
    val latest:StateFlow<LocationSample?>
    val buffer:LocationBuffer
    fun start()
    fun stop()
}
class AndroidLocationProvider(private val context:Context):LocationProvider {
    private val client=LocationServices.getFusedLocationProviderClient(context)
    private val _latest=MutableStateFlow<LocationSample?>(null)
    override val latest:StateFlow<LocationSample?> = _latest
    override val buffer=LocationBuffer()
    private var started=false
    var onRawSample:(LocationSample)->Unit = {}
    private val callback=object:LocationCallback() {
        override fun onLocationResult(result:LocationResult) {
            for(loc in result.locations) {
                if(!loc.hasAccuracy()) continue
                val sample=LocationSample(UUID.randomUUID().toString(),loc.latitude,loc.longitude,loc.accuracy,
                    loc.time,loc.elapsedRealtimeNanos,if(loc.hasAltitude()) loc.altitude else null,
                    if(loc.hasVerticalAccuracy()) loc.verticalAccuracyMeters else null,
                    if(loc.hasSpeed()) loc.speed else null,if(loc.hasSpeedAccuracy()) loc.speedAccuracyMetersPerSecond else null,
                    if(loc.hasBearing()) loc.bearing else null,if(loc.hasBearingAccuracy()) loc.bearingAccuracyDegrees else null,
                    loc.provider,loc.isFromMockProvider)
                onRawSample(sample); buffer.add(sample); _latest.value=sample
            }
        }
    }
    override fun start() {
        if(started) return
        require(ContextCompat.checkSelfPermission(context,Manifest.permission.ACCESS_FINE_LOCATION)==PackageManager.PERMISSION_GRANTED) { "Grant precise location permission." }
        val request=LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY,1000)
            .setMinUpdateIntervalMillis(500).setMaxUpdateDelayMillis(1000).build()
        client.requestLocationUpdates(request,callback,Looper.getMainLooper())
        started=true
    }
    override fun stop() { if(started) client.removeLocationUpdates(callback); started=false }
}
